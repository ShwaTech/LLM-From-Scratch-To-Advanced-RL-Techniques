"""

🧠 1. What Problem Does KV Cache Solve?

In autoregressive LLMs (like GPT, LLaMA, etc.), the model generates one token at a time.
At each new token, it performs self-attention over all previously generated tokens.

Without optimization:
    For token 1 → attends to nothing (just itself).
    For token 2 → attends to token 1.
    For token 3 → attends to tokens 1 and 2.
    ...
    For token n → attends to all 1...n–1.

That means for n tokens, total work ≈ O(n²).
Worse: we recompute attention for all past tokens every step!
That’s extremely inefficient for long contexts.

⚙️ 2. What is KV Cache?
During inference, we don’t need to recompute past attention.
The keys (K) and values (V) from previous tokens never change — we can just store them.
So, we cache them once and reuse them on every new token generation.
Hence: KV Cache = stored Keys and Values from previous steps.

📊 3. How It Works (Step-by-Step)
Let’s say we’re generating text with a transformer.
Step 1: Input: "Once"
    Compute query (Q₁), key (K₁), and value (V₁).
    Store K₁ and V₁ in cache.
Step 2: Next input: "upon"
    Compute new Q₂ (for "upon").
    Load old K₁, V₁ from cache.
    Concatenate them with new K₂, V₂:
        K_total = [K₁, K₂]
        V_total = [V₁, V₂]
    Compute attention(Q₂, K_total, V_total)
    Store new K₂, V₂ to cache.
Step 3: Repeat this for each new token:
    Queries are computed fresh.
    Keys and values are reused.

✅ That reduces redundant computation.

"""


from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class KVCache:
    k: torch.Tensor  # (B, H, T, D)
    v: torch.Tensor  # (B, H, T, D)
    
    @property
    def T(self):
        return self.k.size(2)


class RollingKV:
    """
    Rolling buffer with optional attention sink.
    Keeps first `sink` tokens + last `window` tokens.
    """
    def __init__(self, window: int, sink: int = 0):
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None
    
    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        
        # crop
        if self.k.size(2) > self.window + self.sink:
            sink_part = self.k[:, :, :self.sink, :]
            sink_val  = self.v[:, :, :self.sink, :]
            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]
            self.k = torch.cat([sink_part, tail_k], dim=2)
            self.v = torch.cat([sink_val, tail_v], dim=2)
        
        return self.k, self.v
