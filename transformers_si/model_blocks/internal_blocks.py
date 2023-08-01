import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import Dataset, DataLoader

# from transformers import AutoTokenizer

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import os

from utils.logging import logs

class ScaledDotProductAttention(nn.Module) :

    def __init__(self, 
                 n_heads:int = 8,
                 d_model:int = 512,
                 mask:bool = False
        ) -> None :

        """
        Args:
            n_heads (int): Number of heads in the multi head attention. Defualts to 8
            d_model (int, optional): Dimension of the input. Defaults to 512.
            mask (bool, optional): Whether to apply masking. Defaults to False
        """

        super(ScaledDotProductAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.mask = mask
        self.d_k = int(d_model/n_heads)

    def forward(self,
                key : torch.Tensor,
                query : torch.Tensor,
                value : torch.Tensor
        ) -> torch.Tensor :

        """
        Calculate scaler dot product of key, query and values as described in https://arxiv.org/pdf/1706.03762.pdf

        Args:
            key (torch.Tensor): Key tensor. Shape = (n_heads, batch_size, seq_len, d_model/n_heads)
            query (torch.Tensor): Query tensor. Shape = (n_heads, batch_size, seq_len, d_model/n_heads)
            value (torch.Tensor): Value tensor. Shape = (n_heads, batch_size, seq_len, d_model/n_heads)

        Returns:
            value_with_attention: Value with attention applied. Shape = (n_heads, batch_size, seq_len, d_model/n_heads)
        """

        # assert key.size() == query.size() == value.size(), "Key, query and value must have same shape"

        batch_size, seq_len = key.size(1), key.size(2)

        attention_scores = torch.matmul(query, key.transpose(2, 3))/torch.sqrt(torch.tensor(self.d_k))
        attention_scores = torch.softmax(attention_scores, dim = 3)
        
        if self.mask :
            mask = torch.ones(self.n_heads, batch_size, seq_len, seq_len)
            mask = torch.tril(mask)
            attention_scores = torch.matmul(attention_scores, mask)
            
        value_with_attention = torch.matmul(attention_scores, value)

        return value_with_attention, attention_scores
    

##############################################################################



class MultiHeadAttention(nn.Module) :
    
    def __init__(self, 
                 n_head: int = 8, 
                 d_model: int = 512, 
                 dropout: float = 0.1, 
                 mask: bool = False,
                 self_attention:bool = True
        ) :

        """
        Args:
            n_head (int): Number of heads. Defaults to 8.
            d_model (int): Dimension of input. Defaults to 512.
            dropout (float): Dropout rate. Defaults to 0.1.
            mask (bool): Whether to mask the attention. Defaults to False.
            self_attention (bool): Whether to use self attention. Defaults to True.
        """

        super(MultiHeadAttention, self).__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.self_attention = self_attention

        self.d_k = self.d_v = d_model // n_head
        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v)

        self.attention = ScaledDotProductAttention(n_head, d_model, mask)

        self.mha_linear = nn.Linear(d_model, d_model)

        # nn.init.normal_(self.w_qs.weight, mean = 0, std = np.sqrt(2.0 / (d_model + self.d_k)))
        # nn.init.normal_(self.w_ks.weight, mean = 0, std = np.sqrt(2.0 / (d_model + self.d_k)))
        # nn.init.normal_(self.w_vs.weight, mean = 0, std = np.sqrt(2.0 / (d_model + self.d_v)))

    def forward(self, x, q = None) :

        """
        Implementation of multi head attention layer.

        Args:
            x (torch.Tensor): Padded input with the shaep batch_len, seq_len, d_model
            q (torch.Tensor): Query with the shape batch_size, seq_len, d_model. Defaults to None.
        
        Returns:
            torch.Tensor: Values with multiheadattention applied. Shape = (batch_size, seq_len, d_model)
        
        Raises:
            ValueError: If mode is cross attention and query passed in forward is None.
            ValueError: If mode is cross attention and shape of query is not same as input coming from encoder.
        
        References:
            https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/MultiHeadAttention.py
            https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Transformer.py
            https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/PositionalEncoding.py
        """
         
        if not self.self_attention:
            if q is None :
                raise ValueError("q is required for cross attention")
            # elif x.size() != q.size() :
            #     raise ValueError("q and X must have same size")
        else :
            q = x

        key = F.gelu(self.w_ks(x))
        query = F.gelu(self.w_qs(q))
        value = F.gelu(self.w_vs(x))

        ## keeping n_heads as major dimension
        key = key.view(-1, key.size(0), key.size(1), self.d_k)
        query = query.view(-1, query.size(0), query.size(1), self.d_k)
        value = value.view(-1, value.size(0), value.size(1), self.d_v)

        value, attention = self.attention(key, query, value)

        value = value.view(value.size(1), value.size(2), -1)

        value = self.dropout(value)

        value = F.gelu(self.mha_linear(value))

        return value, attention
    

##############################################################################


class AddLayerNormalization(nn.Module) :

    def __init__(self, d_model) :
        
        super().__init__()

        self.layer_norm = nn.LayerNorm([d_model])

    def forward(self, x, mha_output) :

        return self.layer_norm(x + mha_output)
    

##############################################################################


class PointWiseFeedforward(nn.Module) :

    def __init__(self, 
                 d_ff: int = 2048, 
                 d_model: int = 512
    ) -> None :
        
        """
        Args:
            d_ff (int): Intermediate size of the feedforward layer.
            d_model (int):  Size of the embeddings.
        """
        
        super(PointWiseFeedforward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x) :

        linear1_output = self.linear1(x)
        linear2_output = self.linear2(F.gelu(linear1_output))

        return linear2_output
        