import os
import torch
from ast_hop.compiler import ASTCompiler

def main():
    compiler = ASTCompiler()
    
    # Path to our compiler source file to run the test on
    target_file = "ast_hop/compiler.py"
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found.")
        return
        
    with open(target_file, "r") as f:
        source_code = f.read()
        
    print(f"=== Compiling Source File: {target_file} ===")
    tokens, jump_map = compiler.compile_source(source_code)
    
    total_tokens = len(tokens)
    print(f"Total Token Count: {total_tokens}")
    print(f"Total Block Jumps Found: {len(jump_map) // 2}\n")
    
    print("Identified AST Block Scopes:")
    print(f"{'Token Range':<15} | {'Byte Range':<15} | {'Code Snippet Header'}")
    print("-" * 75)
    
    # We walk the jump map. To prevent duplicates, we only print start -> end (start < end)
    skipped_tokens = 0
    for start, end in sorted(jump_map.items()):
        if start < end:
            # Reconstruct the header from the start token
            header_tokens = tokens[start : min(start + 5, end)]
            header_text = compiler.decompile_tokens(header_tokens).replace("\n", " ").strip()
            
            # Reconstruct byte ranges from offsets
            # Retrieve byte offsets by decoding up to start and end
            token_ids = tokens.tolist()
            start_byte = len(compiler.encoding.decode(token_ids[:start]).encode("utf-8"))
            end_byte = len(compiler.encoding.decode(token_ids[:end]).encode("utf-8"))
            
            print(f"{start:<3} -> {end:<7} | {start_byte:<3} -> {end_byte:<7} | {header_text}...")
            
    # Calculate actual skipped tokens by simulating the forward skimming path
    skipped_tokens = 0
    t = 0
    while t < total_tokens:
        if t in jump_map and t < jump_map[t]:
            end = jump_map[t]
            skipped_tokens += (end - t - 1)
            t = end + 1
        else:
            t += 1
            
    savings_ratio = (skipped_tokens / total_tokens) * 100
    print("-" * 75)
    print(f"Theoretical Prefill Skimming Savings: {skipped_tokens} / {total_tokens} tokens ({savings_ratio:.2f}% savings)")

if __name__ == "__main__":
    main()
