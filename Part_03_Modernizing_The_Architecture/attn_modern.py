"""

💡 Sliding Window Attention is an efficient variant of self-attention where each token only attends to
a fixed number of nearby tokens (a local window) instead of all previous tokens.
This reduces computational and memory complexity from O(n²) to O(n × w),
enabling much longer context lengths. While it captures strong local dependencies and speeds up training
and inference, it can miss long-range relationships — often mitigated by combiningit
with a few global attention tokens or mechanisms like Attention Sinks.

💡 Attention Sinks is a mechanism that introduces special tokens — called “sink tokens” — which
act as global aggregators or information hubs inside a transformer’s attention mechanism.
These tokens (sometimes just one or a few per layer) are trained to absorb
information from all other tokens and optionally broadcast global context back to them.

🧩 Attention sinks are special tokens (or positions) in the attention mechanism that are always
globally attended to — regardless of sliding window limits.

🧩 Attention sinks == the concept (specific tokens or positions that are globally visible).

🧩 Sink tokens act as bridges across the whole text.

💡 Attention Sinks + Sliding-Window Self-Attention — explained clearly
Short version: a sink (aka global / aggregator token) is a small set of special tokens that
connect distant local windows. When used with sliding-window attention, each token attends to a nearby window
plus the sink(s). The sink attends (or is attended to) more broadly, so information can flow
across windows without full O(n²) attention.


💡 What is Grouped Query Attention (GQA)?
Grouped Query Attention simplifies how LLMs understand large amounts of text by bundling similar pieces together. This makes the model faster and smarter, as it can focus on groups of words at a time instead of each word individually.
Grouped Query Attention (GQA) is a method that interpolates between multi-query attention (MQA) and multi-head attention (MHA) in Large Language Models (LLMs). It aims to achieve the quality of MHA while maintaining the speed of MQA.
GQA divides query heads into G groups, each of which shares a single key head and value head.

"""

from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPECache, apply_rope_single
from kv_cache import KVCache  # your existing class



class CausalSelfAttentionModern(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        rope: bool = True,
        max_pos: int = 4096,
        sliding_window: int | None = None,
        attention_sink: int = 0,
        n_kv_head: int | None = None  # ← NEW
    ):  
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head      # ← NEW (GQA defaults to MHA)
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        self.group_size = self.n_head // self.n_kv_head
        self.d_head = n_embd // n_head
        
        # Separate projections for Q vs K/V (sizes differ under GQA)  ← CHANGED
        self.wq = nn.Linear(n_embd, self.n_head * self.d_head, bias=False)
        self.wk = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.use_rope = rope
        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink
    
    
    def _maybe_init_rope(self, device):
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)
    
    
    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """
        x: (B,T,C). If kv_cache given, we assume generation (T small, often 1).
        kv_cache: optional KVCache from previous steps.
        start_pos: position index of first token in x (for RoPE).
        """
        B, T, C = x.shape
        self._maybe_init_rope(x.device)
        
        # Projections
        q = self.wq(x).view(B, T, self.n_head,   self.d_head).transpose(1, 2)    # (B,H, T,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)
        
        # RoPE on *current* tokens (cached keys are already rotated)
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos, sin = self.rope_cache.get(pos)
            q = apply_rope_single(q, cos, sin)   # (B,H, T,D)
            k = apply_rope_single(k, cos, sin)   # (B,Hk,T,D)
        
        # Concatenate past cache (cache is stored in Hk heads)
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk, Tpast+T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v
        
        # Sliding-window + attention-sink (crop along seq length)
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)
        
        # --- GQA expand: repeat K/V heads to match Q heads before attention ---
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
        else:
            k_attn, v_attn = k_all, v_all
        
        # Scaled dot-product attention (PyTorch scales internally)
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(
            q, k_attn, v_attn,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )  # (B,H,T,D)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        
        # Update KV cache (store compact Hk heads, not expanded)
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk,*,D)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        
        new_cache = KVCache(k_new, v_new)
        
        return y, new_cache
