"""

🧩 Context
These two functions — bradley_terry_loss and margin_ranking_loss — are both pairwise ranking losses,
often used in preference learning or reward modeling for fine-tuning LLMs.

Loss	        | Formula	                        | Behavior	                    | Probabilistic?
----------------|-----------------------------------|-------------------------------|------------------
Bradley-Terry	| softplus(-(r_pos - r_neg))	    | Smooth (sigmoid-based)	    | ✅ Yes
Margin Ranking	| max(0, margin - (r_pos - r_neg))	| Hard margin (hinge-based)	    | ❌ No
-------------------------------------------------------------------------------------------------------

💡 Summary
Function	          | Meaning	                                                            | Goal
----------------------|---------------------------------------------------------------------|--------------------------------------------------------
bradley_terry_loss	  | Soft, probabilistic comparison between positive and negative scores	| Increase probability that r_pos > r_neg
----------------------|---------------------------------------------------------------------|--------------------------------------------------------
margin_ranking_loss	  | Hard margin enforcement	                                            | Make sure r_pos is larger than r_neg by at least margin
-----------------------------------------------------------------------------------------------------------------------------------------------------

🧠 Both losses train models to prefer “better” outputs over “worse” ones —
Bradley–Terry does it smoothly (sigmoid-style), while Margin Ranking does it strictly (hinge-style).

"""


from __future__ import annotations
import torch, torch.nn.functional as F


def bradley_terry_loss(r_pos: torch.Tensor, r_neg: torch.Tensor) -> torch.Tensor:
    """
    -log sigma(r_pos - r_neg) = softplus(-(r_pos - r_neg))
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    """
    diff = r_pos - r_neg
    
    return F.softplus(-diff).mean()


def margin_ranking_loss(r_pos: torch.Tensor, r_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
    """
    y = torch.ones_like(r_pos)
    
    return F.margin_ranking_loss(r_pos, r_neg, y, margin=margin)
