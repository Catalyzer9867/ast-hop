# Specification: Tree-Sitter Compiler Pipeline for AST-Hop

This component provides a compiler pipeline that converts raw Python source code into token sequences and bidirectional jump maps using AST block boundaries extracted from Tree-sitter. It also supports parsing token sequences back into valid source strings.

---

## File Manifest

* [ast_hop/compiler.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/ast_hop/compiler.py): Integrates Tree-sitter and tokenization. Maps syntax nodes to token-level index offsets.
* [benchmark/run_compiler_pipeline.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/benchmark/run_compiler_pipeline.py): Script demonstrating the compilation of Python modules and calculating prefill savings ratios.
* [tests/test_compiler.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/tests/test_compiler.py): Unit tests verifying AST parsing, token mappings, round-trip serialization, and jump map correctness.

---

## Data Models & Interfaces

### Tree-Sitter Compiler Loop (`ast_hop/compiler.py`)

```python
import torch
from typing import Dict, List, Tuple

class ASTCompiler:
    def __init__(self, vocab_file: str = None):
        """
        Initializes the compiler, loading the BPE tokenizer and Tree-sitter parser
        for Python.
        """
        pass

    def compile_source(self, source_code: str) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Parses source code to:
        1. Tokenize the text into subword token IDs.
        2. Identify block nodes (functions, classes, branches) in the AST.
        3. Align character offsets of these nodes to token indices.
        4. Construct the bidirectional jump map mapping start token indices to end token indices.
        
        Args:
            source_code: Python code string.
            
        Returns:
            tokens: [seq_len] LongTensor of subword token IDs.
            jump_map: Bidirectional mapping (token_start_index <-> token_end_index).
        """
        pass

    def decompile_tokens(self, tokens: torch.Tensor) -> str:
        """
        Converts a sequence of token IDs back into source code string.
        """
        pass
```

---

## Execution Phases

### Phase 1: Environment Setup and Tree-Sitter Initialization
* Install `tree-sitter` and `tree-sitter-python` packages.
* Set up parser loading inside `ASTCompiler.__init__`.

### Phase 2: Token-to-Byte Alignment
* Tokenize source code while preserving token-to-character byte offset mapping (e.g. using BPE tokenizer offset outputs).
* Map AST node byte boundaries to exact subword token indices.

### Phase 3: AST Block Parsing and Jump Mapping
* Walk the Tree-sitter syntax tree to identify target blocks: `function_definition`, `class_definition`, `if_statement`, `for_statement`, `while_statement`.
* Construct bidirectional index jumps mapping the start token of a block to the end token of that block.
* Verify correct round-trip serialization.

---

## Acceptance Criteria

### Phase 1 & 2: Environment and Alignment Verification
Run pytest verifying successful initialization and token mapping:
```bash
pytest tests/test_compiler.py -k "test_compiler_initialization"
```
* **Success Condition**: Passes with exit code `0`.

### Phase 3: AST-Jump Map Extraction Verification
Run pytest verifying AST block traversal and round-trip alignment:
```bash
pytest tests/test_compiler.py -k "test_compiler_roundtrip"
```
* **Success Condition**: Passes with exit code `0`, confirming:
  1. The compiler successfully parses Python functions and classes.
  2. Jump map contains exact boundary token index offsets.
  3. `decompile_tokens(compile_source(code)[0]) == code` (identity mapping matches).
