import pytest
import torch
import torch.optim as optim
from ast_hop.parser import parse_syntax_blocks
from ast_hop.model import ASTHop
from ast_hop.dataset import generate_dataset, VOCAB
from ast_hop.train import train_epoch

def test_model_shapes():
    vocab_size = len(VOCAB)
    embed_dim = 16
    hidden_dim = 32
    num_classes = 2
    
    model = ASTHop(vocab_size, embed_dim, hidden_dim, num_classes)
    
    tokens = torch.randint(1, vocab_size, (50,))
    jump_map = { 10: 25, 25: 10 }  # Mock jump
    
    logits, visited, log_probs, actions, entropy = model(tokens, jump_map, deterministic=False)
    
    assert logits.shape == (1, num_classes)
    assert len(visited) > 0
    assert entropy.ndim == 0

def test_skimming_performance():
    # Set seed for reproducibility
    torch.manual_seed(42)
    import random
    random.seed(42)
    
    vocab_size = len(VOCAB)
    embed_dim = 16
    hidden_dim = 32
    num_classes = 2
    
    model = ASTHop(vocab_size, embed_dim, hidden_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Generate datasets
    train_data = generate_dataset(250)
    val_data = generate_dataset(50)
    
    # Hyperparameters
    gamma = 0.6  # Reward weight for skipping (balanced to prioritize task accuracy)
    entropy_beta = 0.02
    baseline = 0.0
    
    best_accuracy = 0.0
    best_read_ratio = 1.0
    
    # Train for 25 epochs
    for epoch in range(25):
        metrics, baseline = train_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_data,
            gamma=gamma,
            baseline=baseline,
            entropy_beta=entropy_beta
        )
        
        # Validation evaluation
        model.eval()
        correct = 0
        total_read = 0.0
        total_len = 0.0
        
        with torch.no_grad():
            for tokens, jump_map, target in val_data:
                logits, visited, _, _, _ = model(tokens, jump_map, deterministic=True)
                pred = torch.argmax(logits, dim=-1)
                if pred.item() == target.item():
                    correct += 1
                total_read += len(visited)
                total_len += tokens.size(0)
                
        val_acc = correct / len(val_data)
        val_read_ratio = total_read / total_len
        
        best_accuracy = max(best_accuracy, val_acc)
        best_read_ratio = min(best_read_ratio, val_read_ratio)
        
        # Print progress to stdout for visibility
        print(f"Epoch {epoch+1:02d} | Val Acc: {val_acc:.3f} | Val Read Ratio: {val_read_ratio:.3f} | Train Acc: {metrics['accuracy']:.3f}")
        
    print(f"Best Accuracy: {best_accuracy:.3f}, Best Read Ratio: {best_read_ratio:.3f}")
    
    # Assert validation targets are achieved
    assert best_accuracy >= 0.90, f"Target accuracy of 90% not reached. Best was {best_accuracy:.3f}"
    assert best_read_ratio < 0.30, f"Token reading ratio was not under 30%. Best was {best_read_ratio:.3f}"
