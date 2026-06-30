import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class ASTHop(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_actions: int = 2
    ):
        """
        Args:
            vocab_size: Vocabulary size.
            embed_dim: Token embedding dimension.
            hidden_dim: GRU hidden state dimension.
            num_classes: Target classification classes.
            num_actions: Number of policy actions (e.g. 2 for STEP/SKIP, 3 for STEP/SKIP/SPAWN).
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_cell = nn.GRUCell(embed_dim, hidden_dim)
        
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.predict_head = nn.Linear(hidden_dim, num_classes)
        if num_actions > 2:
            self.generation_head = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        tokens: torch.Tensor,
        jump_map: Dict[int, int],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor]:
        """
        Executes a dynamic skimming forward pass.
        
        Args:
            tokens: [seq_len] Input sequence of token IDs.
            jump_map: Bidirectional mapping of block jumps (index -> index).
            deterministic: If True, selects argmax action. If False, samples from policy.
            
        Returns:
            task_logits: [1, num_classes] Final classification prediction.
            visited_indices: List of token indices processed by the recurrent cell.
            policy_log_probs: [num_decisions] Log probabilities for the taken actions.
            actions_taken: List of binary actions selected (0=STEP, 1=SKIP).
            avg_entropy: [1] The average policy distribution entropy (for regularization).
        """
        seq_len = tokens.size(0)
        device = tokens.device
        
        # Initialize hidden state
        hidden = torch.zeros(1, self.hidden_dim, device=device)
        
        visited_indices = []
        policy_log_probs = []
        actions_taken = []
        policy_entropies = []
        
        t = 0
        while t < seq_len:
            token_id = tokens[t].unsqueeze(0)  # Shape [1]
            x = self.embedding(token_id)        # Shape [1, embed_dim]
            hidden = self.rnn_cell(x, hidden)
            
            visited_indices.append(t)
            
            # Check if t is an opening boundary in the jump map
            if t in jump_map and t < jump_map[t]:
                policy_logits = self.policy_head(hidden)  # [1, 2]
                policy_probs = torch.softmax(policy_logits, dim=-1)
                
                # Calculate entropy: -sum(p * log(p))
                entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-10), dim=-1)
                policy_entropies.append(entropy)
                
                if deterministic:
                    action = torch.argmax(policy_probs, dim=-1)
                else:
                    dist = torch.distributions.Categorical(probs=policy_probs)
                    action = dist.sample()
                
                action_item = action.item()
                log_prob = torch.log(policy_probs[0, action_item] + 1e-10)
                
                policy_log_probs.append(log_prob)
                actions_taken.append(action_item)
                
                if action_item == 1:
                    # Skip block! Jump to end of block
                    t = jump_map[t]
                else:
                    # Action 0 (STEP) or Action 2 (SPAWN) - step inside the block
                    t += 1
            else:
                t += 1
                
        task_logits = self.predict_head(hidden)
        
        if policy_log_probs:
            policy_log_probs_tensor = torch.stack(policy_log_probs)
            avg_entropy = torch.mean(torch.stack(policy_entropies))
        else:
            policy_log_probs_tensor = torch.tensor([], device=device)
            avg_entropy = torch.tensor(0.0, device=device)
            
        return task_logits, visited_indices, policy_log_probs_tensor, actions_taken, avg_entropy
