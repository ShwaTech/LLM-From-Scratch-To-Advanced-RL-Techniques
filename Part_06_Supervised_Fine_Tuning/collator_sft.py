"""

✍️ 1. What Happens During SFT ?
During Supervised Fine-Tuning, you take:
    1. a base pretrained model, and
    2. a dataset of prompt–response pairs (instruction + ideal answer),
and train the model again — but with supervision (like a teacher grading it).

🎯 2. The Objective ?
You train the model so that:
    When it sees the prompt, it predicts the response token by token.
This is just a causal language modeling (next-token prediction) loss, but applied to instruction-formatted data.

⚙️ 3. The Training Process
Input formatting — Combine prompt + response:
    <bos> User: Explain AI simply. <eos>
    Assistant: AI means teaching computers to learn from data. <eos>
Training steps:
1. Tokenize it using the model’s tokenizer.
2. Mask: Only compute loss on the response part (not on the prompt).
3. Compute cross-entropy loss between predicted and true tokens.
4. Backpropagate & update weights.
So the model learns how to respond to prompts like a human tutor would.

💡 4. Why It Matters
Pretrained models just predict text continuation — they don’t “follow instructions.”
SFT teaches them to:
    1. Follow human-style prompts.
    2. Respond in structured, helpful, safe ways.
    3. Understand chat-like input/output patterns.
Essentially, SFT turns a raw LLM into an assistant.

📊 After SFT, the Model Can:
✅ Follow structured instructions
✅ Stay on-topic
✅ Produce coherent responses
✅ Be used as a policy model for PPO, RLHF or GRPO

"""


from __future__ import annotations
from typing import List, Tuple
import torch
import traceback

# Reuse tokenizers: prefer BPE from Part 4 if available; else byte-level from Part 3
import sys
from pathlib import Path as _P

sys.path.append(str(_P(__file__).resolve().parents[1]/'Part_04_Scaling_Up'))
try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False

sys.path.append(str(_P(__file__).resolve().parents[1]/'Part_03_Modernizing_The_Architecture'))
try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

from formatters import Example, format_example, format_prompt_only



class SFTCollator:
    """
    Turn (instruction,response) into token ids and masked labels for causal LM (6.2).
    Labels for the prompt part are set to -100 so they don't contribute to loss.
    """
    def __init__(self, block_size: int = 256, bpe_dir: str | None = None):
        self.block_size = block_size
        self.tok = None
        
        if _HAS_BPE:
            # If a trained tokenizer directory exists from Part 4, you can `load` it.
            # Otherwise we create an ad-hoc BPE on the fly using fallback prompts during demo.
            try:
                self.tok = BPETokenizer(vocab_size=8000)
                if bpe_dir:
                    self.tok.load(bpe_dir)
                    print(f"Loaded BPE tokenizer from {bpe_dir}")
                else:
                    # weak ad-hoc training would belong elsewhere; for the demo we assume Part 4 tokenizer exists
                    pass
            except Exception:
                print(traceback.format_exc())
                self.tok = None
        
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()
        
        if self.tok is None:
            raise RuntimeError("No tokenizer available. Install tokenizers or ensure Part 3 ByteTokenizer exists.")
    
    
    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)
    
    
    def encode(self, text: str) -> List[int]:
        if hasattr(self.tok, 'encode'):
            ids = self.tok.encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        
        # ByteTokenizer-like
        return list(text.encode('utf-8'))
    
    
    def collate(self, batch: List[Tuple[str,str]]):
        # Build "prompt + response" and create label mask where prompt positions are -100.
        input_ids = []
        labels = []
        for prompt, response in batch:
            prefix_text = format_prompt_only(prompt).replace('</s>','')
            text = format_example(Example(prompt, response))
            
            ids = self.encode(text)[:self.block_size]
            prompt_ids = self.encode(prefix_text)[:self.block_size]
            n_prompt = min(len(prompt_ids), len(ids))
            
            x = ids
            y = ids.copy()
            
            for t in range(len(y) - 1):
                y[t] = ids[t + 1]
            
            y[-1] = -100
            
            for i in range(n_prompt-1):
                y[i] = -100
            
            input_ids.append(x)
            labels.append(y)
        
        # pad to block_size
        def pad_to(ids, val):
            if len(ids) < self.block_size:
                ids = ids + [val]*(self.block_size - len(ids))
            
            return ids[:self.block_size]
        
        x = torch.tensor([pad_to(s, 2) for s in input_ids], dtype=torch.long)
        y = torch.tensor([pad_to(s, -100) for s in labels], dtype=torch.long)
        
        return x, y
