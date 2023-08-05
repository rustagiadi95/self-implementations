import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers_fmt.model_blocks.internal_blocks import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    AddLayerNormalization,
    PointWiseFeedforward
)

from utils.logging import logs





###################################### SINGLE ENCODER LAYER ######################################

class EncoderLayer(nn.Module) :

    def __init__(self,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 device: str = 'cuda'
        ) -> None :
        
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(n_heads, d_model, device = device)
        self.layer_norm = AddLayerNormalization(d_model)
        self.pff = PointWiseFeedforward(d_ff, d_model)
        self.layer_norm2 = AddLayerNormalization(d_model)

    def forward(self, x) :

        mha_output, mha_attention_scores = self.mha(x)
        # logs(f"mha_output shape: {mha_output.shape}")
        norm_output1 = self.layer_norm(x, mha_output)
        # logs(f"norm_output1 shape: {norm_output1.shape}")

        pff_output = self.pff(norm_output1)
        # logs(f"pff_output shape: {pff_output.shape}")
        norm_output2 = self.layer_norm2(norm_output1, pff_output)
        # logs(f"norm_output2 shape: {norm_output2.shape}")

        return norm_output2, mha_attention_scores
    



###################################### SINGLE DECODER LAYER ######################################

class DecoderLayer(nn.Module) :

    def __init__(self, 
                 n_heads,
                 d_model,
                 d_ff, 
                 device
    ) -> None :

        super(DecoderLayer, self).__init__()

        self.mha = MultiHeadAttention(n_head=n_heads, d_model=d_model, mask = True, device = device)
        self.cross_mha = MultiHeadAttention(n_head=n_heads, d_model=d_model, self_attention=False, device = device)
        self.layer_norm1 = AddLayerNormalization(d_model)
        self.layer_norm2 = AddLayerNormalization(d_model)
        self.layer_norm3 = AddLayerNormalization(d_model)
        self.pff = PointWiseFeedforward(d_ff, d_model)

    def forward(self, x, enc_out) :
        ## passing encoder output to all decoder layers : to be discussed with Deepak
        decoder_query, _ = self.mha(x)
        norm_decoder_query = self.layer_norm1(x, decoder_query)

        x, _ = self.cross_mha(enc_out, norm_decoder_query)
        norm_cross_x = self.layer_norm2(norm_decoder_query, x)

        x = self.pff(norm_cross_x)
        norm_decoder_output = self.layer_norm3(norm_cross_x, x)

        return norm_decoder_output
    
