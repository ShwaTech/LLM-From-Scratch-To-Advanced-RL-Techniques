"""

Relative Position Embeddings help models understand relationships between tokens based on
how far apart they are — enabling better generalization, longer context reasoning, and faster convergence.

Types of Relative Position Embeddings
There are a few variants across transformer models:

Type       	                          | Example Models	           | Description
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Additive RPE	                      | Transformer-XL, T5	       | Adds a relative bias term depending on distance (i–j).
Rotary Position Embedding (RoPE)	  | GPT-NeoX, LLaMA, Mistral   | Rotates queries & keys in embedding space based on their relative distance. Very efficient and elegant.
ALiBi (Attention with Linear Biases)  | LLaMA 3, MPT	           | Adds linear bias directly to attention scores, scaling with distance — simple and scalable for long contexts.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Why It’s Better

✅ Generalizes to longer sequences than trained on.
✅ Captures relative order instead of fixed position indices.
✅ Improves translation & reasoning (important for long context and multi-turn understanding).
✅ Efficient — some forms (like RoPE or ALiBi) add almost no extra parameters.


=========================
---- Our Focus: RoPE ----
=========================
🧠 LLMs rely on Rotary Position Embeddings (RoPE) to understand the relative position of words within a sequence.
Each word in the sequence is assigned a unique embedding based on its position.
This embedding is calculated using a combination of sine and cosine functions,
incorporating its distance from the beginning and end of the sequence.
However, standard RoPE struggles with longer sequences than those encountered during training.
The embedding values for distant words become very similar, making it
difficult for the LLM to distinguish their relative positions.
Therefore, the embeddings become less effective, leading to poor performance. 
Extrapolation essentially refers to the maximum sequence length an LLM can handle effectively with its original RoPE settings.
Beyond this limit, performance degrades significantly.

🧠 RoPE Scaling modifies the RoPE calculations to improve the model's ability to handle longer sequences.
The core idea is to tweak the base value used in the RoPE calculations.
This value controls the rate at which the sine and cosine functions oscillate,
affecting the embedding distribution. Increasing the base value can spread out the embeddings,
making them more distinct for longer sequences. While decreasing it can introduce periodicity,
allowing the model to handle longer sequences that wrap around this cycle.

"""


from __future__ import annotations
import torch
import math


class RoPECache:
    """
    RoPE (Rotary Positional Encoding) cache.
    Precompute cos/sin for positions up to max_pos for even head_dim.
    """
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self.device = device
        self._build(max_pos)
    
    
    def get(self, positions: torch.Tensor):
        # positions: (T,) or (1,T)
        if positions.dim() == 2:
            positions = positions[0]
        
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        
        if need > self.max_pos:
            # grow tables
            self._build(max(need, int(self.max_pos * 2)))
        
        cos = self.cos[positions]  # (T, D/2)
        sin = self.sin[positions]
        
        return cos, sin
    
    
    def _build(self, max_pos: int):
        """
        (Re)build cos/sin tables for a new max_pos.
        """
        self.max_pos = max_pos
        
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        
        t = torch.arange(max_pos, device=self.device).float()
        
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
        
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)


def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotate pairs along last dim for RoPE.
    x: (B,H,T,D) with D even; cos/sin: (T,D/2)
    """
    assert x.size(-1) % 2 == 0
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    
    out = torch.empty_like(x)
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    
    return out
