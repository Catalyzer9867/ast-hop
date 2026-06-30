import torch
from typing import Dict, List, Tuple
from tree_sitter import Language, Parser
import tree_sitter_python
import tiktoken

class ASTCompiler:
    def __init__(self):
        """
        Initializes the compiler, loading the tiktoken BPE tokenizer and Tree-sitter parser
        for Python.
        """
        # Load tiktoken BPE tokenizer
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Load Tree-sitter Python parser
        lang = Language(tree_sitter_python.language())
        self.parser = Parser(lang)
        
        # AST Node types that represent navigable code blocks
        self.block_types = {
            "function_definition",
            "class_definition",
            "if_statement",
            "for_statement",
            "while_statement"
        }

    def compile_source(self, source_code: str) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Parses source code to:
        1. Tokenize the text into subword token IDs.
        2. Identify block nodes in the AST.
        3. Align byte offsets of these nodes to token indices.
        4. Construct the bidirectional jump map.
        
        Returns:
            tokens: [seq_len] LongTensor of subword token IDs.
            jump_map: Bidirectional mapping (token_start_index <-> token_end_index).
        """
        # Encode source to bytes and get token byte ranges
        source_bytes = source_code.encode("utf-8")
        token_ids = self.encoding.encode(source_code)
        
        # Reconstruct byte offsets for tokens
        offsets: List[Tuple[int, int]] = []
        current_byte = 0
        for token_id in token_ids:
            token_bytes = self.encoding.decode_bytes([token_id])
            start = current_byte
            end = start + len(token_bytes)
            offsets.append((start, end))
            current_byte = end
            
        # Parse syntax tree using tree-sitter
        tree = self.parser.parse(source_bytes)
        root_node = tree.root_node
        
        # Collect all block nodes
        block_nodes = []
        def walk(node):
            if node.type in self.block_types:
                block_nodes.append(node)
            for child in node.children:
                walk(child)
                
        walk(root_node)
        
        # Map byte offsets to token indices
        def byte_to_token_idx(byte_idx: int) -> int:
            for idx, (start, end) in enumerate(offsets):
                if start <= byte_idx < end:
                    return idx
            # Fallback to nearest boundary
            if byte_idx <= 0:
                return 0
            return len(offsets) - 1
            
        jump_map: Dict[int, int] = {}
        for node in block_nodes:
            start_token = byte_to_token_idx(node.start_byte)
            # Use node.end_byte - 1 to find the token containing the end token
            end_token = byte_to_token_idx(max(node.start_byte, node.end_byte - 1))
            
            # Only map if they are different tokens and no start index conflict exists
            if start_token < end_token and start_token not in jump_map and end_token not in jump_map:
                jump_map[start_token] = end_token
                jump_map[end_token] = start_token
                
        return torch.tensor(token_ids, dtype=torch.long), jump_map

    def decompile_tokens(self, tokens: torch.Tensor) -> str:
        """
        Converts a sequence of token IDs back into source code string.
        """
        token_list = tokens.tolist()
        return self.encoding.decode(token_list)
