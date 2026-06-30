from typing import Dict, List

def parse_syntax_blocks(tokens: List[str]) -> Dict[int, int]:
    """
    Scans a token sequence and returns a bi-directional index map 
    linking opening bracket/block indices to their corresponding closing indices.
    
    Args:
        tokens: List of string tokens representing code or symbols.
        
    Returns:
        A dictionary mapping:
            opening_index -> closing_index
            closing_index -> opening_index
    """
    jump_map: Dict[int, int] = {}
    stack = []
    
    open_brackets = { '(': ')', '[': ']', '{': '}' }
    close_brackets = { ')': '(', ']': '[', '}': '{' }
    
    for i, token in enumerate(tokens):
        if token in open_brackets:
            stack.append((token, i))
        elif token in close_brackets:
            if stack:
                top_char, top_idx = stack[-1]
                if open_brackets[top_char] == token:
                    stack.pop()
                    jump_map[top_idx] = i
                    jump_map[i] = top_idx
            # Unmatched closing brackets are ignored to maintain robust parsing.
            
    return jump_map
