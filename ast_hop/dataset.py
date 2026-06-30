import random
import torch
from typing import List, Tuple, Dict
from ast_hop.parser import parse_syntax_blocks

VOCAB = {
    '<PAD>': 0,
    '(': 1, ')': 2,
    '[': 3, ']': 4,
    '{': 5, '}': 6,
    'N': 7,
    'a': 8, 'b': 9, 'c': 10
}

def generate_sample(num_blocks: int = 5, needle_prob: float = 0.5) -> Tuple[List[str], int]:
    """
    Generates a sequence of tokens representing balanced nested scopes.
    '{ }' scopes are noise blocks that contain NO needles and are safe to skip.
    '( )' and '[ ]' scopes may contain the target needle 'N'.
    
    Returns:
        tokens: List of string tokens.
        label: 1 if the needle 'N' is present, 0 otherwise.
    """
    tokens = []
    has_needle = random.random() < needle_prob
    needle_placed = False
    
    for i in range(num_blocks):
        block_type = random.choice(['skip', 'read_parenthesis', 'read_bracket'])
        if block_type == 'skip':
            tokens.append('{')
            length = random.randint(15, 25)
            for _ in range(length):
                tokens.append(random.choice(['a', 'b', 'c']))
            tokens.append('}')
        elif block_type == 'read_parenthesis':
            tokens.append('(')
            length = random.randint(3, 8)
            place_here = has_needle and not needle_placed and random.random() < 0.6
            for _ in range(length):
                if place_here and not needle_placed:
                    tokens.append('N')
                    needle_placed = True
                    place_here = False
                else:
                    tokens.append(random.choice(['a', 'b', 'c']))
            tokens.append(')')
        else:
            tokens.append('[')
            length = random.randint(3, 8)
            place_here = has_needle and not needle_placed and random.random() < 0.6
            for _ in range(length):
                if place_here and not needle_placed:
                    tokens.append('N')
                    needle_placed = True
                    place_here = False
                else:
                    tokens.append(random.choice(['a', 'b', 'c']))
            tokens.append(']')
            
    # Guarantee needle is placed if has_needle was selected
    if has_needle and not needle_placed:
        tokens.append('(')
        tokens.append('N')
        tokens.append(')')
        needle_placed = True
        
    label = 1 if needle_placed else 0
    return tokens, label

def get_token_ids(tokens: List[str]) -> torch.Tensor:
    return torch.tensor([VOCAB[t] for t in tokens], dtype=torch.long)

def generate_dataset(num_samples: int) -> List[Tuple[torch.Tensor, Dict[int, int], torch.Tensor]]:
    dataset = []
    for _ in range(num_samples):
        tokens, label = generate_sample()
        token_ids = get_token_ids(tokens)
        jump_map = parse_syntax_blocks(tokens)
        target = torch.tensor([label], dtype=torch.long)
        dataset.append((token_ids, jump_map, target))
    return dataset
