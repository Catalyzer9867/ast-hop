import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from ast_hop.model import ASTHop

def train_step(
    model: ASTHop,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    jump_map: Dict[int, int],
    target: torch.Tensor,
    gamma: float,
    baseline: float,
    entropy_beta: float
) -> Tuple[float, float, float, float, float, int]:
    """
    Performs a single parameter update step.
    
    Returns:
        loss_val, task_loss_val, reward_val, read_ratio_val, entropy_val, action_taken_count
    """
    optimizer.zero_grad()
    
    # Forward pass
    task_logits, visited_indices, policy_log_probs, actions_taken, avg_entropy = model(
        tokens=tokens,
        jump_map=jump_map,
        deterministic=False
    )
    
    # Task loss (cross-entropy)
    loss_task = nn.functional.cross_entropy(task_logits, target)
    
    # Reward: -loss_task + gamma * (percentage of tokens skipped)
    seq_len = tokens.size(0)
    tokens_read = len(visited_indices)
    read_ratio = tokens_read / float(seq_len)
    saved_ratio = 1.0 - read_ratio
    
    reward = -loss_task.item() + gamma * saved_ratio
    
    # Policy loss using REINFORCE with baseline subtraction
    loss_policy = torch.tensor(0.0, device=tokens.device)
    if len(policy_log_probs) > 0:
        advantage = reward - baseline
        loss_policy = -torch.sum(policy_log_probs * advantage)
        
    # Total loss: task_loss + policy_loss - entropy_regularization
    loss_total = loss_task + loss_policy - entropy_beta * avg_entropy
    
    loss_total.backward()
    
    # Gradient clipping to stabilize training
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return (
        loss_total.item(),
        loss_task.item(),
        reward,
        read_ratio,
        avg_entropy.item(),
        len(actions_taken)
    )

def train_epoch(
    model: ASTHop,
    optimizer: torch.optim.Optimizer,
    data_loader: List[Tuple[torch.Tensor, Dict[int, int], torch.Tensor]],
    gamma: float,
    baseline: float,
    entropy_beta: float
) -> Tuple[Dict[str, float], float]:
    """
    Trains the model for one epoch over the dataset.
    """
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_reward = 0.0
    total_read_ratio = 0.0
    total_entropy = 0.0
    correct = 0
    count = 0
    
    for tokens, jump_map, target in data_loader:
        # Run step
        loss, task_loss, reward, read_ratio, entropy, action_count = train_step(
            model=model,
            optimizer=optimizer,
            tokens=tokens,
            jump_map=jump_map,
            target=target,
            gamma=gamma,
            baseline=baseline,
            entropy_beta=entropy_beta
        )
        
        # Update baseline (moving average)
        baseline = 0.9 * baseline + 0.1 * reward
        
        # Log evaluation predictions
        model.eval()
        with torch.no_grad():
            task_logits, _, _, _, _ = model(tokens, jump_map, deterministic=True)
            pred = torch.argmax(task_logits, dim=-1)
            if pred.item() == target.item():
                correct += 1
        model.train()
        
        total_loss += loss
        total_task_loss += task_loss
        total_reward += reward
        total_read_ratio += read_ratio
        total_entropy += entropy
        count += 1
        
    metrics = {
        "loss": total_loss / count,
        "task_loss": total_task_loss / count,
        "reward": total_reward / count,
        "read_ratio": total_read_ratio / count,
        "entropy": total_entropy / count,
        "accuracy": correct / count
    }
    
    return metrics, baseline
