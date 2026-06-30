import torch
import concurrent.futures
from typing import Dict, List, Tuple, Callable

class RecursiveAgent:
    def __init__(self, model, hidden_dim: int = 384, gating_alpha: float = 0.5):
        """
        Args:
            model: ASTHop model instance.
            hidden_dim: GRU hidden state dimension.
            gating_alpha: Weight given to parent's hidden state when merging child states.
        """
        self.model = model
        self.hidden_dim = hidden_dim
        self.gating_alpha = gating_alpha

    def execute_skimming_pass(
        self,
        tokens: torch.Tensor,
        jump_map: Dict[int, int],
        initial_hidden: torch.Tensor = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        Walks the token list, spawning nested subagents on Action 2 (SPAWN).
        
        Returns:
            final_hidden: [1, hidden_dim] Final aggregated context state.
            visited_indices: List of token indices processed.
            actions_taken: List of actions selected.
        """
        device = tokens.device
        seq_len = tokens.size(0)
        
        # Initialize hidden state
        if initial_hidden is not None:
            hidden = initial_hidden.clone()
        else:
            hidden = torch.zeros(1, self.hidden_dim, device=device)
            
        visited_indices = []
        actions_taken = []
        
        t = 0
        while t < seq_len:
            token_id = tokens[t].unsqueeze(0)
            x = self.model.embedding(token_id)
            hidden = self.model.rnn_cell(x, hidden)
            visited_indices.append(t)
            
            # Check if t is an opening boundary in the jump map
            if t in jump_map and t < jump_map[t]:
                with torch.no_grad():
                    policy_logits = self.model.policy_head(hidden)
                    policy_probs = torch.softmax(policy_logits, dim=-1)
                
                if deterministic:
                    action = torch.argmax(policy_probs, dim=-1).item()
                else:
                    dist = torch.distributions.Categorical(probs=policy_probs)
                    action = dist.sample().item()
                    
                actions_taken.append(action)
                
                if action == 1:
                    # Action 1 (SKIP): Jump directly to the end
                    t = jump_map[t]
                elif action == 2:
                    # Action 2 (SPAWN): Spawn a subagent to process the block slice in parallel
                    block_start = t + 1
                    block_end = jump_map[t]
                    
                    if block_start < block_end:
                        block_tokens = tokens[block_start:block_end]
                        # Construct a shifted local jump map for the slice
                        sub_jump_map = {
                            k - block_start: v - block_start
                            for k, v in jump_map.items()
                            if block_start <= k < block_end and block_start <= v <= block_end
                        }
                        
                        # Execute subagent in a parallel thread pool executor
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                self._spawn_child,
                                block_tokens,
                                sub_jump_map,
                                hidden,
                                deterministic
                            )
                            child_hidden, child_visited = future.result()
                            
                        # Merge the subagent's state back
                        hidden = self.gating_alpha * hidden + (1 - self.gating_alpha) * child_hidden
                        visited_indices.extend([idx + block_start for idx in child_visited])
                        
                    t = jump_map[t]
                else:
                    # Action 0 (STEP): Process sequentially
                    t += 1
            else:
                t += 1
                
        return hidden, visited_indices, actions_taken

    def _spawn_child(
        self,
        tokens: torch.Tensor,
        jump_map: Dict[int, int],
        parent_hidden: torch.Tensor,
        deterministic: bool
    ) -> Tuple[torch.Tensor, List[int]]:
        """Helper to run a subagent instance."""
        subagent = RecursiveAgent(self.model, self.hidden_dim, self.gating_alpha)
        child_hidden, child_visited, _ = subagent.execute_skimming_pass(
            tokens=tokens,
            jump_map=jump_map,
            initial_hidden=parent_hidden,
            deterministic=deterministic
        )
        return child_hidden, child_visited

    def execute_generation_pass(
        self,
        prompt_tokens: torch.Tensor,
        syntax_mask_fn: Callable[[List[int]], torch.Tensor] = None,
        max_tokens: int = 128
    ) -> torch.Tensor:
        """
        Generates a sequence of token IDs autoregressively.
        """
        device = prompt_tokens.device
        generated = prompt_tokens.tolist()
        
        # Initialize state with prompt prefix
        hidden = torch.zeros(1, self.hidden_dim, device=device)
        for tok in prompt_tokens:
            token_id = tok.unsqueeze(0)
            x = self.model.embedding(token_id)
            hidden = self.model.rnn_cell(x, hidden)
            
        for _ in range(max_tokens):
            # Predict next token distribution
            logits = self.model.generation_head(hidden)
            
            # Apply syntax logit masking if provided
            if syntax_mask_fn is not None:
                mask = syntax_mask_fn(generated)
                logits = logits + mask.to(device)
                
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()
            
            # Stop token generation boundary (0 is padding/stop token in BPE vocabs)
            if next_token == 0:
                break
                
            generated.append(next_token)
            
            # Feed back next token
            token_tensor = torch.tensor([next_token], device=device)
            x = self.model.embedding(token_tensor)
            hidden = self.model.rnn_cell(x, hidden)
            
        return torch.tensor(generated, device=device)
