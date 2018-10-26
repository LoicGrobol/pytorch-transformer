import math

import torch
import torch.jit

import numpy as np


@torch.jit.script
def gelu(x):
    """gelu(input) -> Tensor

    Applies the gaussian error linear unit function element-wise. See
    :class:`~GeLU` for more details
    """
    return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * torch.pow(x, 3))))


class GELU(torch.jit.ScriptModule):
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

    def __init__(self, sublayer, dimension, dropout):
        super(ResidualConnection, self).__init__()
        self.sublayer = sublayer
        self.norm = torch.nn.LayerNorm(dimension)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(self.sublayer(x, *args, **kwargs)))


class TransformerFeedForward(torch.jit.ScriptModule):
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


@torch.jit.script
def scaled_dot_product_attention(query, key, value, mask=None):
    """Apply the scaled dot-product attention.

    See :class:`~ScaledDotProductAttention` for more details.
    """
    # OPTIMISE: this is comparable to transpose+matmul in terms of speed for
    # now but it should get better
    scores = torch.einsum('...ij,...kj->...ik', (query, key))/math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill_(mask, -np.inf)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn, value)


class ScaledDotProductAttention(torch.jit.ScriptModule):
    """Apply the scaled dot-product attention

    Inputs: query, key, value, mask
        - **query** of shape `(*, sequence_length, request_size)`
        - **key** of shape  `(*, sequence_length, request_size)`
        - **value** of shape  `(*, sequence_length, features_size)`
        - **mask** :class:`torch.ByteTensor` of shape  `(*, sequence_length)
          with `1`s on the sequence items to mask (either for padding or
          masking)

    Output: `(*, sequence_length, features_size)`
    """

    def forward(self, query, key, value, mask=None):
        return scaled_dot_product_attention(query, key, value, mask)


class MultiHeadedAttention(torch.jit.ScriptModule):
    def __init__(self, features_dim, n_heads):
        super(MultiHeadedAttention, self).__init__()

        self.features_dim = features_dim
        self.n_heads = n_heads
        self.heads_dim = features_dim // n_heads

        self.query_projectors = torch.nn.Linear(self.features_dim, self.n_heads*self.heads_dim)
        self.key_projectors = torch.nn.Linear(self.features_dim, self.n_heads*self.heads_dim)
        self.value_projectors = torch.nn.Linear(self.features_dim, self.n_heads*self.heads_dim)

        self.output_linear = torch.nn.Linear(self.n_heads*self.heads_dim, self.features_dim)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.query_projectors(query).reshape(
            batch_size, -1, self.n_heads, self.heads_dim
        ).transpose(1, 2)
        key = self.key_projectors(query).reshape(
            batch_size, -1, self.n_heads, self.heads_dim
        ).transpose(1, 2)
        value = self.value_projectors(query).reshape(
            batch_size, -1, self.n_heads, self.heads_dim
        ).transpose(1, 2)

        x, attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.head_dims)

        return self.output_linear(x)


class MultiHeadedSelfAttention(MultiHeadedAttention):
    def forward(self, inpt, mask):
        return super()(inpt, inpt, inpt, mask)


class TransformerBlock(torch.nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, input_dim, output_dim, attn_heads, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedSelfAttention(features_size=input_dim, n_heads=attn_heads)
        self.feed_forward = TransformerFeedForward(
            input_dim=input_dim,
            hidden_dim=output_dim,
            dropout=dropout,
        )
        self.atention_sublayer = ResidualConnection(
            self.attention,
            dimension=input_dim,
            dropout=dropout,
        )
        self.feed_forward_sublayer = ResidualConnection(
            self.feed_forward,
            dimension=input_dim,
            dropout=dropout,
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.attention_sublayer(self.dropout(x), mask=mask)
        x = self.feed_forward_sublayer(x)
        return self.dropout(x)


class PositionalEmbeddings(torch.jit.ScriptModule):
    "Add positional embeddings to a sequence"

    def __init__(self, dimension, max_len=1024):
        super(PositionalEmbeddings, self).__init__()
        self.weigth = torch.nn.Parameter(
            torch.empty(dimension, max_len)
        )
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, inpt):
        seq_len = inpt.size(1)
        return inpt + self.weight.narrow(1, 0, seq_len)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension=512,
        blocks=6,
        hidden_dimension=None,
        max_len=1024,
        dropout=0.1,
        attn_heads=8,
    ):
        super(Encoder, self).__init__()
        if hidden_dimension is None:
            hidden_dimension = output_dimension
        self.positional_embeddings = PositionalEmbeddings(input_dimension, max_len)
        self.input_block = TransformerBlock(input_dimension, hidden_dimension, attn_heads, dropout)
        self.output_block = TransformerBlock(
            hidden_dimension, output_dimension, attn_heads, dropout
        )
        self.blocks = torch.nn.ModuleList([
            self.input_block,
            *(
                TransformerBlock(hidden_dimension, hidden_dimension, attn_heads, dropout)
                for _ in range(blocks-2)
            ),
            self.output_block,
        ])

    def forward(self, inpt, mask):
        out = inpt
        for b in self.block:
            out = b(out, mask)
        return out
