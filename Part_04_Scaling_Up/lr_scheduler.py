"""

🚀 Warmup + Cosine Learning Rate Scheduler Explained
The Problem It Solves:
Training large models (like Transformers) with AdamW or similar optimizers is sensitive to the learning rate.

If you start with a large LR immediately, early gradients can explode or destabilize training.

So instead of jumping to the full LR on step 1, we:
    1. Gradually increase LR during a warmup period.
    2. Then gradually decrease LR following a smooth cosine decay.

That’s the Warmup + Cosine schedule.

Phase	        | Description	                         | Benefit
----------------|----------------------------------------|--------------------------------------
Warmup	        | Gradually increases LR from 0 → max	 | Prevents instability in early steps
Cosine Decay	| Smoothly decreases LR to 0	         | Stabilizes convergence
Combined	    | “WarmupCosineLR”	                     | Gold-standard scheduler for LLMs
------------------------------------------------------------------------------------------------

"""

import math


class WarmupCosineLR:
    """
    Linear warmup → cosine decay (per-step API).
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps+1, total_steps)
        self.base_lr = base_lr
        self.step_num = 0
    
    
    def step(self):
        self.step_num += 1
        
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        
        return lr
