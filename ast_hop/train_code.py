import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from ast_hop.compiler import ASTCompiler
from ast_hop.model import ASTHop
from ast_hop.train import train_epoch

def find_python_files(root_dir: str) -> List[str]:
    py_files = []
    for root, dirs, files in os.walk(root_dir):
        if ".venv" in root or "__pycache__" in root or ".git" in root or ".pytest_cache" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

def load_code_dataset(compiler: ASTCompiler) -> Tuple[List[Tuple[torch.Tensor, Dict[int, int], torch.Tensor]], int]:
    py_files = find_python_files(".")
    print(f"Scanning codebase. Found {len(py_files)} Python files to tokenize.")
    
    raw_samples = []
    all_token_ids = set()
    
    for file_path in py_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        tokens, jump_map = compiler.compile_source(content)
        all_token_ids.update(tokens.tolist())
        
        # Label: 1 if file defines a class, 0 otherwise
        label = 1 if "class " in content else 0
        raw_samples.append((tokens.tolist(), jump_map, label, file_path))
        
    # Build token remapping
    unique_tokens = sorted(list(all_token_ids))
    token_to_local = {tok_id: idx + 1 for idx, tok_id in enumerate(unique_tokens)}
    vocab_size = len(unique_tokens) + 1
    print(f"Codebase unique token count: {len(unique_tokens)} (Local Vocab Size: {vocab_size})")
    
    dataset = []
    for token_list, jump_map, label, file_path in raw_samples:
        local_tokens = torch.tensor([token_to_local[t] for t in token_list], dtype=torch.long)
        target = torch.tensor([label], dtype=torch.long)
        dataset.append((local_tokens, jump_map, target))
        
    return dataset, vocab_size

def run_code_training(epochs: int = 40) -> Tuple[ASTHop, Dict[str, float]]:
    compiler = ASTCompiler()
    dataset, vocab_size = load_code_dataset(compiler)
    
    # Scale model dimensions
    embed_dim = 128
    hidden_dim = 384
    num_classes = 2
    
    model = ASTHop(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Scaled AST-Hop Parameter Count: {total_params:,} parameters.")
    
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    gamma = 0.5
    entropy_beta = 0.02
    baseline = 0.0
    
    best_accuracy = 0.0
    best_read_ratio = 1.0
    
    # Duplicate dataset slightly to ensure batch stability during training epochs
    train_loader = dataset * 5
    
    for epoch in range(epochs):
        metrics, baseline = train_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            gamma=gamma,
            baseline=baseline,
            entropy_beta=entropy_beta
        )
        
        # Validation on actual original set
        model.eval()
        correct = 0
        total_read = 0
        total_tokens = 0
        
        with torch.no_grad():
            for tokens, jump_map, target in dataset:
                logits, visited, _, _, _ = model(tokens, jump_map, deterministic=True)
                pred = torch.argmax(logits, dim=-1)
                if pred.item() == target.item():
                    correct += 1
                total_read += len(visited)
                total_tokens += tokens.size(0)
                
        val_acc = correct / len(dataset)
        val_read_ratio = total_read / total_tokens
        
        best_accuracy = max(best_accuracy, val_acc)
        best_read_ratio = min(best_read_ratio, val_read_ratio)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {metrics['loss']:.4f} | Val Acc: {val_acc:.2%} | Val Read Ratio: {val_read_ratio:.2%}")
            
    print(f"\nFinal Convergence Metrics:")
    print(f"Best Accuracy: {best_accuracy:.2%}")
    print(f"Best Prefill Read Ratio: {best_read_ratio:.2%}")
    
    return model, {"accuracy": best_accuracy, "read_ratio": best_read_ratio}

if __name__ == "__main__":
    run_code_training()
