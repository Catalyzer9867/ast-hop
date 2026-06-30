import argparse
import sys
import os
import torch
from ast_hop.compiler import ASTCompiler
from ast_hop.model import ASTHop
from ast_hop.agent.sandbox import CodeSandbox
from ast_hop.agent.recursive_agent import RecursiveAgent

def main():
    parser = argparse.ArgumentParser(description="Hop-Agent: Recursive Multi-Agent Coder CLI")
    parser.add_argument("--task", type=str, required=True, help="Refactoring or coding task instruction")
    parser.add_argument("--dir", type=str, required=True, help="Path to the target codebase directory")
    parser.add_argument("--test-cmd", type=str, required=True, help="Shell command to run the test suite")
    parser.add_argument("--model-path", type=str, default=None, help="Optional path to model checkpoint")
    
    args = parser.parse_args()
    
    print(f"[*] Initializing Hop-Agent on codebase: {args.dir}")
    if not os.path.exists(args.dir):
        print(f"[!] Error: Target directory does not exist: {args.dir}")
        sys.exit(1)
        
    # Compile the codebase files
    compiler = ASTCompiler()
    all_tokens = []
    global_jump_map = {}
    
    # Scan python files
    py_files = [
        os.path.join(dp, f) for dp, dn, fn in os.walk(args.dir) for f in fn if f.endswith(".py")
    ]
    
    if not py_files:
        print("[!] Error: No python files found in directory.")
        sys.exit(1)
        
    print(f"[*] Found {len(py_files)} Python modules. Compiling AST jump maps...")
    
    token_offset = 0
    for py_file in py_files:
        try:
            with open(py_file, "r") as f:
                source = f.read()
            tokens, jump_map = compiler.compile_source(source)
            
            # Merge to global token stream
            all_tokens.extend(tokens)
            for k, v in jump_map.items():
                global_jump_map[k + token_offset] = v + token_offset
            token_offset += len(tokens)
        except Exception as e:
            print(f"[!] Skipping {py_file} due to compilation error: {str(e)}")
            
    print(f"[*] Codebase compilation complete: {len(all_tokens)} total BPE tokens.")
    
    # Initialize/load model
    device = torch.device("cpu")
    vocab_size = 16384  # default vocab size for 300M pretraining
    hidden_dim = 384
    embed_dim = 128
    
    # If a checkpoint is provided, load its configuration
    if args.model_path and os.path.exists(args.model_path):
        print(f"[*] Loading model checkpoint from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        # Handle checkpoint mapping
        vocab_size = checkpoint.get("vocab_size", vocab_size)
        hidden_dim = checkpoint.get("hidden_dim", hidden_dim)
        embed_dim = checkpoint.get("embed_dim", embed_dim)
        
        model = ASTHop(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_actions=3  # 3 actions: STEP, SKIP, SPAWN
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        print("[*] Initializing fresh local AST-Hop model...")
        model = ASTHop(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_actions=3
        )
        
    model.eval()
    
    # Run sandbox test check
    print(f"[*] Executing sandbox verification runner...")
    sandbox = CodeSandbox(args.dir)
    success, output = sandbox.execute_test(args.test_cmd)
    if success:
        print("[+] Base test suite passed successfully.")
    else:
        print(f"[-] Base test suite failed. Traceback extracted:\n{output}")
        
    # Execute the agentic skimming pass
    print("[*] Launching Recursive Multi-Agent Skimming & Routing...")
    agent = RecursiveAgent(model, hidden_dim=hidden_dim)
    
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
    
    final_hidden, visited, actions = agent.execute_skimming_pass(
        tokens=tokens_tensor,
        jump_map=global_jump_map,
        deterministic=True
    )
    
    spawns = actions.count(2)
    skips = actions.count(1)
    reads = len(visited)
    savings = (1.0 - (reads / len(all_tokens))) * 100 if all_tokens else 0.0
    
    print("[+] Execution Finished:")
    print(f"    - Visited tokens: {reads} / {len(all_tokens)} ({savings:.2f}% savings)")
    print(f"    - Subagent Spawns: {spawns} parallel threads launched")
    print(f"    - Skipped Blocks: {skips} blocks skipped completely")

if __name__ == "__main__":
    main()
