"""

The Swish activation function is similar in shape to ReLU, as it's close to zero when x is negative,
and close to x when x is positive.
It is defined as Swish(x) = x * sigmoid(ßx), where ß is a learnable parameter.
The cool thing about having a learnable parameter in the activation function is that the network learns how to better shape it.

GLU stands for Gated Linear Unit, which is a rather complicated activation function
consisting of two linear transformations, one of them activated by sigmoid,
so that GLU(x) = (Wx+b)*sigmoid(Vx+c).

SwiGLU is just a portmanteau of Swish and GLU, and as such,
it's simply a GLU that is activated using the Swish function instead of a sigmoid,
so that SwiGLU(x) = (Wx+b)*Swish(Vx+c).

"""


import torch.nn as nn

class SwiGLU(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) activation function.
    SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = mult * dim
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        a = self.w1(x)
        b = self.act(self.w2(x))
        return self.drop(self.w3(a * b))
