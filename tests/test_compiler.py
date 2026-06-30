import pytest
import torch
from ast_hop.compiler import ASTCompiler

def test_compiler_initialization():
    compiler = ASTCompiler()
    assert compiler is not None
    assert compiler.encoding is not None
    assert compiler.parser is not None

def test_compiler_roundtrip():
    compiler = ASTCompiler()
    
    source_code = """def calculate_sum(a, b):
    result = a + b
    if result > 0:
        return result
    else:
        return 0

class Calculator:
    def __init__(self):
        self.value = 0
"""
    
    # Run compiler
    tokens, jump_map = compiler.compile_source(source_code)
    
    # Verify shape and type
    assert isinstance(tokens, torch.Tensor)
    assert tokens.ndim == 1
    assert len(jump_map) > 0
    
    # Verify bidirectional mappings
    for k, v in jump_map.items():
        assert jump_map[v] == k
        
    # Verify round-trip decompression
    reconstructed = compiler.decompile_tokens(tokens)
    assert reconstructed == source_code
    
    # Verify that the function definition is in the jump map
    # Token 0 is 'def', which should be mapped to the end of the function
    assert 0 in jump_map
    end_token_of_def = jump_map[0]
    assert end_token_of_def > 0
