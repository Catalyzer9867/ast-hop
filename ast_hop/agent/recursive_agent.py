import torch
import concurrent.futures
from typing import Dict, List, Tuple, Callable

class RecursiveAgent:
    def __init__(self, model, hidden_dim: int = 384, gating_alpha: float = 0.5, use_qwen: bool = False, qwen_model: str = "qwen2.5-coder:1.5b-instruct"):
        """
        Args:
            model: ASTHop model instance.
            hidden_dim: GRU hidden state dimension.
            gating_alpha: Weight given to parent's hidden state when merging child states.
            use_qwen: Whether to route code generation to local Qwen model.
            qwen_model: Model name for local Ollama instance.
        """
        self.model = model
        self.hidden_dim = hidden_dim
        self.gating_alpha = gating_alpha
        self.use_qwen = use_qwen
        self.qwen_model = qwen_model
        
        import tiktoken
        self.encoding = tiktoken.get_encoding("cl100k_base")

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

    def _generate_via_qwen(self, prompt: str) -> str:
        """Calls the local Ollama Qwen instance to generate code."""
        import requests
        url = "http://localhost:11434/api/generate"
        system_prompt = (
            "You are an expert software developer. Write clean, complete, and correct code "
            "based on the prompt and context provided. Keep the code simple, clean, and self-contained. "
            "Do NOT import external HTTP or networking libraries (such as aiohttp or requests) "
            "unless the prompt explicitly asks for web or API access. "
            "Output ONLY the code inside the file. Do NOT wrap it in markdown code blocks and do NOT output any explanations."
        )
        payload = {
            "model": self.qwen_model,
            "prompt": f"{system_prompt}\n\nTask: {prompt}\n\nCode:",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"\n[!] Ollama returned error status {response.status_code}: {response.text}")
                return ""
        except Exception as e:
            print(f"\n[!] Failed to connect to local Ollama server: {e}")
            return ""

    def execute_generation_pass(
        self,
        prompt_tokens: torch.Tensor,
        syntax_mask_fn: Callable[[List[int]], torch.Tensor] = None,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Generates a sequence of token IDs autoregressively or via local Qwen.
        """
        device = prompt_tokens.device
        
        if self.use_qwen:
            # 1. Decode prompt tokens to text
            prompt_text = self.encoding.decode(prompt_tokens.tolist())
            
            # 2. Generate via local Qwen API call
            generated_code = self._generate_via_qwen(prompt_text)
            if generated_code:
                # Clean up markdown code block fences if Qwen wrapped them
                generated_code = generated_code.strip()
                if generated_code.startswith("```"):
                    first_newline = generated_code.find("\n")
                    if first_newline != -1:
                        generated_code = generated_code[first_newline + 1:]
                    else:
                        generated_code = generated_code[3:]
                if generated_code.endswith("```"):
                    generated_code = generated_code[:-3]
                generated_code = generated_code.strip()
                
                # Encode response back into tokens
                gen_tokens = self.encoding.encode(generated_code)
                return torch.tensor(gen_tokens, device=device)
            else:
                print("[*] Falling back to local ASTHop generation head...")
                
        # Fallback to local GRU model generation head
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
                
            if temperature > 0.0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs[0], num_samples=1).item()
            else:
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

    def determine_filename(self, prompt: str) -> str:
        """Calls local Qwen to determine the best filename for a given prompt."""
        if not self.use_qwen:
            return "agent_output.py"
            
        import requests
        url = "http://localhost:11434/api/generate"
        system_prompt = (
            "You are an assistant that determines the best python filename for a given task. "
            "Output ONLY the filename, with a .py extension, and absolutely nothing else. "
            "Example: for 'write a calculator', return 'calculator.py'."
        )
        payload = {
            "model": self.qwen_model,
            "prompt": f"{system_prompt}\n\nTask: {prompt}\n\nFilename:",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 15
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                fn = response.json().get("response", "").strip()
                # Strip backticks, quotes, spaces
                fn = fn.replace("`", "").replace("'", "").replace('"', "").strip()
                if not fn.endswith(".py"):
                    # Extract the first word if it returned more text
                    words = fn.split()
                    if words:
                        fn = words[0]
                    if not fn.endswith(".py"):
                        fn += ".py"
                return fn
            return "agent_output.py"
        except Exception:
            return "agent_output.py"
