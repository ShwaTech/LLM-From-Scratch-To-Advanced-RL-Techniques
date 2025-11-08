from __future__ import annotations
import torch, torch.nn as nn
import sys
from pathlib import Path as _P

# Try user’s structure first (same as before)
sys.path.append(str(_P(__file__).resolve().parents[1]/'Part_03_Modernizing_The_Architecture'))
try:
    from model_utils.model_modern import GPTModern  # type: ignore   ## user-custom path
except Exception:
    from model_modern import GPTModern  # fallback



class PolicyWithoutValue(nn.Module):
    """
    Policy network for GRPO (Group Relative Policy Optimization).
    Unlike PPO, GRPO does NOT use a value head or critic.
    It only produces logits for action probabilities.
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer=4,
        n_head=4,
        n_embd=256,
        use_rmsnorm=True,
        use_swiglu=True,
        rope=True,
        dropout=0.0
    ):
        super().__init__()
        
        # Core policy model (language model)
        self.lm = GPTModern(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            use_rmsnorm=use_rmsnorm,
            use_swiglu=use_swiglu,
            rope=rope,
            dropout=dropout,
        )
    
    
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        """
        Forward pass through the policy (language model).
        Returns:
            logits: (B, T, V) — token-level probabilities.
            loss: optional cross-entropy (if y provided).
        """
        logits, loss, _ = self.lm(x, y)
        
        return logits, loss  # No value output
    
    
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        Generate text samples using the LM’s sampling strategy.
        """
        return self.lm.generate(*args, **kwargs)
