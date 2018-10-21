import torch

import numpy as np


def gelu(x):
    """gelu(input) -> Tensor

    Applies the gaussian error linear unit function element-wise. See
    :class:`~GeLU` for more details
    """
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * torch.pow(x, 3))))


class GELU(torch.nn.Module):
    r"""Applies the gaussian error linear unit function element-wise as
    described in the paper [Bridging nonlinearities and stochastic regularizers
    with Gaussian error linear units](https://arxiv.org/abs/1606.08415)

    Formally, :math:`\text{GELU}(x)=xP(X⩽x)` where :math:`X~\mathcal{N}(0, 1)`

    Here, we use the “slower but more accurate approximation” suggesteed by
    [the reference implementation](https://github.com/hendrycks/GELUs).

    :math:`0.5×x×(1+\text{tanh}(x×0.7978845608(1 + 0.044715x²)))`

    Shape:
        - Input: :math:`(N, *)` where `*` means “any number of additional
          dimensions”
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, x):
        return gelu(x)


class ResidualConnection(torch.nn.Module):
    """Wrap a layer to apply dropout, residual input connection and layer
    normalization
    """

    def __init__(self, sublayer, shape, dropout):
        super(ResidualConnection, self).__init__()
        self.sublayer = sublayer
        self.norm = torch.nn.LayerNorm(shape)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(self.sublayer(x)))


class TransformerFeedForward(torch.nn.Module):
    """The feed-forward sublayer of the transformer block

    To be precise, this is a two-layer feed-forward neural network whose input
    and output dimensions are equal, the default activation is a GELU,
    following the BERT version of the Transformer
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1, activation=None):
        super(TransformerFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.w_2 = torch.nn.Linear(hidden_dim, input_dim)
        self.dropout = torch.nn.Dropout(dropout)
        if activation is None:
            self.activation = GELU()
        else:
            self.activation = activation

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


def scaled_dot_product_attention(query, key, value, mask=None):
    scores = torch.tensordot(query, key, dims=([-1], [-1]))/np.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill_(mask, -np.inf)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, value)


class Attention(torch.nn.Module):
    """Apply the scaled dot-product attention"""

    def forward(self, query, key, value, mask=None):
        return scaled_dot_product_attention(query, key, value, mask)
