# Specification: AST-Hop (Causal Syntactic Skimming Generator)

An architecture designed to bypass the Causal No-Skimming Theorem on nested code syntax. It couples a recurrent sequence model with a syntax parser to enable safe, block-level jumping during prompt prefilling and grammar-constrained autoregressive generation.

---

## File Manifest

* [ast_hop/parser.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/ast_hop/parser.py): Lightweight syntactic boundary parser. Maps matching brackets and block scopes (start index <-> end index).
* [ast_hop/model.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/ast_hop/model.py): Core neural architecture containing the embedding layers, GRU recurrent cell, prediction heads, and the custom dynamic AST-skimming forward loop.
* [ast_hop/train.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/ast_hop/train.py): Policy gradient (REINFORCE) training script implementing joint syntax prediction loss, action entropy penalties, and step-wise token saving rewards.
* [ast_hop/dataset.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/ast_hop/dataset.py): Generator for bracket nested structures (Dyck-N) interleaved with noisy filler blocks to evaluate skipping efficiency.
* [tests/test_ast_hop.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/tests/test_ast_hop.py): Suite of unit and integration tests confirming parsing correctness, shape tracking, and reinforcement training sanity.

---

## Data Models & Interfaces

### Syntax Parser Interface (`ast_hop/parser.py`)

```python
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
    pass
```

### AST-Hop Neural Model (`ast_hop/model.py`)

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

class ASTHop(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int
    ):
        """
        Args:
            vocab_size: Vocabulary size.
            embed_dim: Token embedding dimension.
            hidden_dim: GRU hidden state dimension.
            num_classes: Target classification classes.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_cell = nn.GRUCell(embed_dim, hidden_dim)
        
        # Binary policy head: 
        # Action 0 (STEP): read next sequential token
        # Action 1 (SKIP): jump directly to end of current block
        self.policy_head = nn.Linear(hidden_dim, 2)
        self.predict_head = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        tokens: torch.Tensor,
        jump_map: Dict[int, int],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, List[int], List[torch.Tensor], List[int]]:
        """
        Executes a dynamic skimming forward pass.
        
        Args:
            tokens: [seq_len] Input sequence of token IDs.
            jump_map: Bidirectional mapping of block jumps (index -> index).
            deterministic: If True, selects argmax action. If False, samples from policy.
            
        Returns:
            task_logits: [num_classes] Final classification prediction.
            visited_indices: List of token indices processed by the recurrent cell.
            policy_log_probs: List of log probabilities for the taken actions.
            actions_taken: List of binary actions selected (0=STEP, 1=SKIP).
        """
        pass
```

---

## Execution Phases

### Phase 1: Syntactic Parser and Block Mapping
* Implement `parse_syntax_blocks` in [ast_hop/parser.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/ast_hop/parser.py).
* Add logic to parse nested brackets (`( )`, `[ ]`, `{ }`) and Python-style indentation blocks.
* Add unit tests verifying matching index pairs.

### Phase 2: Core Model and Dynamic Traversal Loop
* Implement `ASTHop` network structure in [ast_hop/model.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/ast_hop/model.py).
* Define the sequential execution engine: if the model is at index $i$ and selects `SKIP` (Action 1) AND index $i$ is in `jump_map`, the pointer updates to `jump_map[i]`. Otherwise, it advances to $i + 1$.
* Verify recurrent state updates correctly digest target entry boundaries.

### Phase 3: REINFORCE Training Loop with Baseline
* Implement training policy gradient in [ast_hop/train.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/ast_hop/train.py).
* Formulate the step reward: 
  $$R = -\mathcal{L}_{\text{task}} + \gamma \cdot \frac{\text{Tokens Saved}}{\text{Total Tokens}}$$
  where $\gamma$ balances parsing accuracy and computational speed.
* Implement moving baseline subtraction to stabilize policy updates.

### Phase 4: Dyck-N Skipping Evaluation
* Create a dataset builder in [ast_hop/dataset.py](file:///Users/ibrahimawad/Desktop/AiKnowledgeBase/Projects/codeArchitecture/ast_hop/dataset.py) generating sequences with valid outer brackets filled with large blocks of random symbol noise that can be syntactically skipped.
* Validate model performance targets.

---

## Acceptance Criteria

### Phase 1: Parser Verification
Run a verification script testing nested scopes:
```bash
python3 -c "from ast_hop.parser import parse_syntax_blocks; map_res = parse_syntax_blocks(['(', 'x', ')']); assert map_res[0] == 2; assert map_res[2] == 0; print('Parser check passed')"
```
* **Success Condition**: Outputs `Parser check passed` with exit code `0`.

### Phase 2: Model Execution & Shape Verification
Run pytest verifying single-step shapes and traversal:
```bash
pytest tests/test_ast_hop.py -k "test_model_shapes"
```
* **Success Condition**: Passes with exit code `0`.

### Phase 3 & 4: Functional Verification
Run convergence tests:
```bash
pytest tests/test_ast_hop.py -k "test_skimming_performance"
```
* **Success Condition**: Passes with exit code `0`, confirming:
  1. Task prediction accuracy is >= 90% on nested evaluation sets.
  2. Active prefill token processing index count ($N_s$) is strictly < 30% of total context length ($L$).

---

## Frontier Model Benchmark Targets (Phase 5)

To evaluate AST-Hop against frontier models on consumer hardware, we target the following metrics:

### 1. Prefill FLOP-Efficiency
* **Goal**: Measure Accuracy-per-Prefill-FLOP on long-context (128k+) token retrieval tasks.
* **Benchmark Script**: `python benchmark/run_frontier_comparison.py --metric flop_efficiency`
* **Success Condition**: Accuracy-per-Prefill-FLOP is > 1000x higher than Gemini 3.5 Flash baseline.

### 2. Syntactical Compilation Rate
* **Goal**: Verify 100% syntactical correctness during autoregressive generation of deeply nested blocks.
* **Benchmark Script**: `python benchmark/run_frontier_comparison.py --metric syntax_correctness`
* **Success Condition**: Syntactical parser/compile success rate is exactly 100%, outperforming frontier model baselines.

### 3. VRAM Footprint Scaling
* **Goal**: Demonstrate constant VRAM footprint scaling up to 1M context tokens.
* **Benchmark Script**: `python benchmark/run_frontier_comparison.py --metric vram_scaling`
* **Success Condition**: Peak VRAM remains under 50MB across all context lengths (32k to 1M tokens), while the frontier model OOMs or scales linearly on consumer hardware.
