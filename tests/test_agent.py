import pytest
import torch
import os
import sys
import tempfile
from ast_hop.model import ASTHop
from ast_hop.agent.sandbox import CodeSandbox
from ast_hop.agent.recursive_agent import RecursiveAgent

def test_sandbox_execution():
    """Verify that sandbox successfully runs command and captures tracebacks."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a dummy python script and test file
        calc_file = os.path.join(tmp_dir, "calc.py")
        with open(calc_file, "w") as f:
            f.write("def add(x, y):\n    return x + y\n")
            
        test_file = os.path.join(tmp_dir, "test_calc.py")
        with open(test_file, "w") as f:
            f.write("from calc import add\ndef test_add():\n    assert add(1, 2) == 4\n")
            
        sandbox = CodeSandbox(tmp_dir)
        
        # Test 1: Run passing command
        success, traceback = sandbox.execute_test("echo 'Running sandbox test'")
        assert success is True
        assert traceback == ""
        
        # Test 2: Run failing test command (should capture failure output)
        success, traceback = sandbox.execute_test(f"{sys.executable} -m pytest {test_file}")
        assert success is False
        assert "FAILURES" in traceback or "AssertionError" in traceback or "assert 3 == 4" in traceback

def test_recursive_agent_spawning():
    """Verify that agent runs sequential traversal, spawns subagents, and merges hidden states."""
    # Model parameters
    vocab_size = 100
    embed_dim = 16
    hidden_dim = 32
    
    # Initialize model with 3 actions (STEP, SKIP, SPAWN)
    model = ASTHop(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_actions=3
    )
    
    # Mock tokens and jump map
    tokens = torch.randint(1, vocab_size, (50,))
    # Create a block spanning from index 10 to 30
    jump_map = {10: 30}
    
    agent = RecursiveAgent(model, hidden_dim=hidden_dim, gating_alpha=0.5)
    
    # Force the policy head weights to output a SPAWN action (Action 2)
    # The policy head projection has shape [3, hidden_dim]
    # We set the bias of index 2 (SPAWN) to a large value so it is always chosen
    with torch.no_grad():
        model.policy_head.bias.fill_(0.0)
        model.policy_head.bias[2] = 100.0  # highly favor SPAWN
        
    hidden, visited, actions = agent.execute_skimming_pass(
        tokens=tokens,
        jump_map=jump_map,
        deterministic=True
    )
    
    # Visited tokens should include indexes outside block (0-9), index 10,
    # the subagent's indexes (11-29), and post-block indexes (30-49)
    assert 10 in visited
    assert 11 in visited
    assert 29 in visited
    assert len(visited) == 50
    assert 2 in actions  # Action 2 (SPAWN) should have been triggered

def test_recursive_agent_skipping():
    """Verify that agent skips block when policy head favors Action 1 (SKIP)."""
    vocab_size = 100
    embed_dim = 16
    hidden_dim = 32
    
    model = ASTHop(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_actions=3
    )
    
    tokens = torch.randint(1, vocab_size, (50,))
    jump_map = {10: 30}
    
    agent = RecursiveAgent(model, hidden_dim=hidden_dim, gating_alpha=0.5)
    
    # Force the policy head to output a SKIP action (Action 1)
    with torch.no_grad():
        model.policy_head.bias.fill_(0.0)
        model.policy_head.bias[1] = 100.0  # highly favor SKIP
        
    hidden, visited, actions = agent.execute_skimming_pass(
        tokens=tokens,
        jump_map=jump_map,
        deterministic=True
    )
    
    # Index 10 is visited (checks skip policy), then jumps directly to 30.
    # Indices 11 to 29 should NOT be visited.
    assert 10 in visited
    for idx in range(11, 30):
        assert idx not in visited
    assert 30 in visited
    assert 1 in actions

def test_generation_pass():
    """Verify that autoregressive generation runs and terminates."""
    vocab_size = 100
    embed_dim = 16
    hidden_dim = 32
    
    model = ASTHop(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_actions=3
    )
    
    agent = RecursiveAgent(model, hidden_dim=hidden_dim)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    
    # Define a syntax mask function that forbids token 0 (stop token) until step 5
    def syntax_mask_fn(history):
        mask = torch.zeros(vocab_size)
        if len(history) < 8:
            mask[0] = -1e9  # ban stop token
        return mask
        
    generated = agent.execute_generation_pass(
        prompt_tokens=prompt,
        syntax_mask_fn=syntax_mask_fn,
        max_tokens=15
    )
    
    assert len(generated) > len(prompt)
    assert generated[0].item() == 1
    assert generated[1].item() == 2
    assert generated[2].item() == 3
