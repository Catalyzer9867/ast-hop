import argparse
import sys
import os
import time
import torch
import threading
from typing import Dict, List
from ast_hop.compiler import ASTCompiler
from ast_hop.model import ASTHop
from ast_hop.agent.sandbox import CodeSandbox
from ast_hop.agent.recursive_agent import RecursiveAgent

# ANSI Color Codes for Premium Terminal Aesthetics
CLR_HEADER = "\033[95m"
CLR_CYAN = "\033[1;36m"
CLR_GREEN = "\033[1;32m"
CLR_YELLOW = "\033[1;33m"
CLR_RED = "\033[1;31m"
CLR_GRAY = "\033[90m"
CLR_RESET = "\033[0m"

# Beautiful Terminal Banner
BANNER = fr"""
{CLR_CYAN}    ___   _____ ______         __  __           
   /   | / ___//_  __/        / / / /___  ____  
  / /| | \__ \  / /  ______  / /_/ / __ \/ __ \\ 
 / ___ |___/ / / /  /_____/ / __  / /_/ / /_/ / 
/_/  |_/____/ /_/          /_/ /_/\\____/ / .___/  {CLR_GRAY}v0.1.0{CLR_CYAN}
                                      /_/       
{CLR_GRAY}Recursive Multi-Agent Coder • Offline • 300M Param Ready{CLR_RESET}
"""

class TerminalSpinner:
    """A thread-safe terminal spinner for showcasing agent processing states."""
    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.spin_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._spin)
        self._thread.daemon = True
        self._thread.start()

    def _spin(self):
        idx = 0
        while self.running:
            sys.stdout.write(f"\r{CLR_CYAN}{self.spin_chars[idx]} {self.message}...{CLR_RESET}")
            sys.stdout.flush()
            idx = (idx + 1) % len(self.spin_chars)
            time.sleep(0.08)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write("\r\033[K")  # Clear the line
        sys.stdout.flush()

def compile_workspace(workspace_dir: str, compiler: ASTCompiler) -> tuple:
    """Scans and compiles files in the workspace directory."""
    all_tokens = []
    global_jump_map = {}
    py_files = []
    
    for dp, dn, fn in os.walk(workspace_dir):
        # Skip cache, build, and virtual environments
        if any(ignored in dp for ignored in [".venv", "__pycache__", "build", "dist", ".git"]):
            continue
        for f in fn:
            if f.endswith(".py"):
                py_files.append(os.path.join(dp, f))
                
    token_offset = 0
    for py_file in py_files:
        try:
            with open(py_file, "r") as f:
                source = f.read()
            tokens, jump_map = compiler.compile_source(source)
            all_tokens.extend(tokens)
            for k, v in jump_map.items():
                global_jump_map[k + token_offset] = v + token_offset
            token_offset += len(tokens)
        except Exception:
            pass
            
    return all_tokens, global_jump_map, py_files

def run_interactive_loop(workspace_dir: str, test_cmd: str, model_path: str = None):
    """Launches the interactive terminal chat REPL."""
    print(BANNER)
    print(f"[*] Initializing workspace: {CLR_CYAN}{os.path.abspath(workspace_dir)}{CLR_RESET}")
    
    compiler = ASTCompiler()
    all_tokens, global_jump_map, py_files = compile_workspace(workspace_dir, compiler)
    
    print(f"[*] Indexed {CLR_GREEN}{len(py_files)}{CLR_RESET} Python modules ({len(all_tokens)} total BPE tokens).")
    
    # Initialize model
    device = torch.device("cpu")
    vocab_size = 100277
    hidden_dim = 384
    embed_dim = 128
    
    if model_path and os.path.exists(model_path):
        print(f"[*] Loading model parameters from: {CLR_YELLOW}{model_path}{CLR_RESET}")
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint.get("vocab_size", vocab_size)
        hidden_dim = checkpoint.get("hidden_dim", hidden_dim)
        embed_dim = checkpoint.get("embed_dim", embed_dim)
        model = ASTHop(vocab_size, embed_dim, hidden_dim, 2, num_actions=3)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        print(f"[*] Initializing local AST-Hop model ({CLR_CYAN}300M architecture configured{CLR_RESET})...")
        model = ASTHop(vocab_size, embed_dim, hidden_dim, 2, num_actions=3)
        
    model.eval()
    agent = RecursiveAgent(model, hidden_dim=hidden_dim)
    sandbox = CodeSandbox(workspace_dir)
    
    print(f"\n{CLR_GREEN}[+] Hop-Agent is ready. Type {CLR_CYAN}/help{CLR_RESET} for commands or ask a question.\n")
    
    while True:
        try:
            # Interactive Prompt Symbol
            user_input = input(f"{CLR_CYAN}hop-agent ❯{CLR_RESET} ").strip()
            if not user_input:
                continue
                
            if user_input.startswith("/"):
                # Handle Commands
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                
                if cmd == "/exit" or cmd == "/quit":
                    print(f"\n{CLR_GRAY}Shutting down agent session. Goodbye!{CLR_RESET}")
                    break
                elif cmd == "/help":
                    print(f"\n{CLR_HEADER}Available Interactive Commands:{CLR_RESET}")
                    print(f"  {CLR_CYAN}/help{CLR_RESET}                       - Show this help message")
                    print(f"  {CLR_CYAN}/files{CLR_RESET}                      - List compiled files in target workspace")
                    print(f"  {CLR_CYAN}/tests{CLR_RESET}                      - Execute verification tests in the sandbox")
                    print(f"  {CLR_CYAN}/skim <task>{CLR_RESET}                - Run a skimming simulation on the workspace")
                    print(f"  {CLR_CYAN}/write <file> <prompt>{CLR_RESET}     - Generate and write code physically to disk")
                    print(f"  {CLR_CYAN}/exit{CLR_RESET}                       - Terminate the chatbot session\n")
                elif cmd == "/files":
                    print(f"\n{CLR_HEADER}Workspace Files Indexed:{CLR_RESET}")
                    for idx, py_file in enumerate(py_files):
                        rel_path = os.path.relpath(py_file, workspace_dir)
                        print(f"  [{idx}] {CLR_GRAY}{rel_path}{CLR_RESET}")
                    print()
                elif cmd == "/tests":
                    spinner = TerminalSpinner("Running Sandbox Tests")
                    spinner.start()
                    success, report = sandbox.execute_test(test_cmd)
                    spinner.stop()
                    if success:
                        print(f"\n{CLR_GREEN}✓ Verification Successful: All tests passed.{CLR_RESET}\n")
                    else:
                        print(f"\n{CLR_RED}✗ Verification Failed. Sandbox test traceback:{CLR_RESET}")
                        print(f"{CLR_GRAY}{report}{CLR_RESET}\n")
                elif cmd == "/skim":
                    task_text = cmd_parts[1] if len(cmd_parts) > 1 else "default_task"
                    spinner = TerminalSpinner("Scanning AST Structure")
                    spinner.start()
                    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
                    final_hidden, visited, actions = agent.execute_skimming_pass(
                        tokens=tokens_tensor,
                        jump_map=global_jump_map,
                        deterministic=True
                    )
                    spinner.stop()
                    
                    spawns = actions.count(2)
                    skips = actions.count(1)
                    reads = len(visited)
                    savings = (1.0 - (reads / len(all_tokens))) * 100 if all_tokens else 0.0
                    
                    print(f"\n{CLR_GREEN}AST-Hop Skimming Analysis for:{CLR_RESET} '{task_text}'")
                    print(f"  • Token Savings:  {CLR_CYAN}{savings:.2f}%{CLR_RESET} ({reads} read, {len(all_tokens)} total)")
                    print(f"  • Subagent Spawns: {CLR_CYAN}{spawns}{CLR_RESET} parallel scopes initiated")
                    print(f"  • Block Skips:     {CLR_CYAN}{skips}{CLR_RESET} structures skipped completely\n")
                elif cmd == "/write":
                    if len(cmd_parts) < 2:
                        print(f"{CLR_RED}Usage: /write <file_path> <prompt_text>{CLR_RESET}\n")
                        continue
                    args_str = cmd_parts[1].split(maxsplit=1)
                    if len(args_str) < 2:
                        print(f"{CLR_RED}Usage: /write <file_path> <prompt_text>{CLR_RESET}\n")
                        continue
                    file_path = os.path.join(workspace_dir, args_str[0])
                    prompt_text = args_str[1]
                    
                    spinner = TerminalSpinner(f"Generating and writing to {args_str[0]}")
                    spinner.start()
                    
                    # 1. Encode prompt
                    prompt_tokens = compiler.encoding.encode(prompt_text)
                    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
                    
                    # 2. Autoregressively generate code tokens using RecursiveAgent
                    generated_tensor = agent.execute_generation_pass(
                        prompt_tokens=prompt_tensor,
                        max_tokens=150
                    )
                    
                    # 3. Decode generated tokens back into clean text
                    generated_code = compiler.encoding.decode(generated_tensor.tolist())
                    
                    # 4. Write string content directly to target file path on disk
                    try:
                        with open(file_path, "w") as f:
                            f.write(generated_code)
                        success_write = True
                    except Exception as e:
                        success_write = False
                        write_err = str(e)
                        
                    spinner.stop()
                    
                    if success_write:
                        print(f"\n{CLR_GREEN}[+] Successfully created and wrote to {args_str[0]}{CLR_RESET}")
                        # Auto-trigger sandbox tests to verify correctness
                        print(f"[*] Running sandbox verification tests...")
                        success, report = sandbox.execute_test(test_cmd)
                        if success:
                            print(f"{CLR_GREEN}✓ Verification Successful: All tests passed on edited file.{CLR_RESET}\n")
                        else:
                            print(f"{CLR_RED}✗ Verification Failed. Sandbox traceback output:{CLR_RESET}")
                            print(f"{CLR_GRAY}{report}{CLR_RESET}\n")
                    else:
                        print(f"\n{CLR_RED}[!] Error writing to file: {write_err}{CLR_RESET}\n")
                else:
                    print(f"{CLR_RED}Unknown command: {cmd}. Type /help for assistance.{CLR_RESET}")
            else:
                # Default behavior: run code generation and write to disk
                # Check if the first word is a target python file name
                words = user_input.split(maxsplit=1)
                if len(words) >= 1 and words[0].endswith(".py"):
                    target_filename = words[0]
                    prompt_text = words[1] if len(words) > 1 else target_filename
                else:
                    target_filename = "agent_output.py"
                    prompt_text = user_input
                    
                file_path = os.path.join(workspace_dir, target_filename)
                
                spinner = TerminalSpinner(f"Generating and writing to {target_filename}")
                spinner.start()
                
                # 1. Encode prompt
                prompt_tokens = compiler.encoding.encode(prompt_text)
                prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
                
                # 2. Autoregressively generate code tokens using RecursiveAgent
                generated_tensor = agent.execute_generation_pass(
                    prompt_tokens=prompt_tensor,
                    max_tokens=150
                )
                
                # 3. Decode generated tokens back into clean text
                generated_code = compiler.encoding.decode(generated_tensor.tolist())
                
                # 4. Write string content directly to target file path on disk
                try:
                    with open(file_path, "w") as f:
                        f.write(generated_code)
                    success_write = True
                except Exception as e:
                    success_write = False
                    write_err = str(e)
                    
                spinner.stop()
                
                if success_write:
                    print(f"\n{CLR_GREEN}[+] Successfully created and wrote to {target_filename}{CLR_RESET}")
                    # Auto-trigger sandbox tests to verify correctness
                    print(f"[*] Running sandbox verification tests...")
                    success, report = sandbox.execute_test(test_cmd)
                    if success:
                        print(f"{CLR_GREEN}✓ Verification Successful: All tests passed.{CLR_RESET}\n")
                    else:
                        print(f"{CLR_RED}✗ Verification Failed. Sandbox traceback output:{CLR_RESET}")
                        print(f"{CLR_GRAY}{report}{CLR_RESET}\n")
                else:
                    print(f"\n{CLR_RED}[!] Error writing to file: {write_err}{CLR_RESET}\n")
                
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{CLR_GRAY}Session interrupted. Goodbye!{CLR_RESET}")
            break

def main():
    parser = argparse.ArgumentParser(description="Hop-Agent: Interactive Multi-Agent Coder")
    parser.add_argument("--task", type=str, default=None, help="One-shot refactoring task (if not provided, launches interactive chat)")
    parser.add_argument("--dir", type=str, default=".", help="Path to the target codebase directory")
    parser.add_argument("--test-cmd", type=str, default="pytest", help="Shell command to run the test suite")
    parser.add_argument("--model-path", type=str, default=None, help="Optional path to model checkpoint")
    
    args = parser.parse_args()
    
    if args.task:
        # Run one-shot CLI execution
        print(f"[*] Executing task: {args.task}")
        compiler = ASTCompiler()
        all_tokens, global_jump_map, py_files = compile_workspace(args.dir, compiler)
        sandbox = CodeSandbox(args.dir)
        
        # Initialize/load model
        device = torch.device("cpu")
        vocab_size = 100277  # default vocab size for cl100k_base tokenizer
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
        agent = RecursiveAgent(model, hidden_dim=384)
        
        tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
        final_hidden, visited, actions = agent.execute_skimming_pass(
            tokens=tokens_tensor,
            jump_map=global_jump_map,
            deterministic=True
        )
        
        print(f"[+] Task skimming analysis complete. Visited {len(visited)}/{len(all_tokens)} tokens.")
        success, report = sandbox.execute_test(args.test_cmd)
        if success:
            print("[+] Sandbox verification test passed.")
        else:
            print(f"[-] Sandbox verification test failed:\n{report}")
    else:
        # Launch interactive REPL loop
        run_interactive_loop(args.dir, args.test_cmd, args.model_path)

if __name__ == "__main__":
    main()
