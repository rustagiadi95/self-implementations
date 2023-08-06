import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers_fmt.model_blocks.transformer_layer import EncoderLayer, DecoderLayer

from utils.logging import logs




###################################### POSITION EMBEDDING ######################################

class PositionEmbedding(nn.Module) :

    def __init__(self,
        max_seq_len: int = 128, 
        d_model: int = 512,
        dropout: int = 0.1,
        device: str = 'cuda'
    ) :

        super(PositionEmbedding, self).__init__()

        self.embedding = torch.zeros(max_seq_len, d_model).to(device)
        self.dropout = nn.Dropout(dropout)
        
        for i in range(max_seq_len) :
            self.embedding[i, 0::2] = torch.sin((i/1000**(2*torch.arange(d_model)[::2]/d_model)))
            self.embedding[i, 1::2] = torch.cos((i/1000**(2*torch.arange(d_model)[1::2]/d_model)))

    def forward(self, x) :

        embedding = torch.repeat_interleave(self.embedding.unsqueeze(0), x.size(0), 0)

        return self.dropout(x + embedding[:, :x.size(1), :])




###################################### ENCODER ######################################

class Encoder(nn.Module) :

    def __init__(self,
                 n_layer: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 device: str = 'cuda'
    ) :
        super(Encoder, self).__init__()
        
        self.encoder = nn.ModuleDict({
            f'encoder_layer_{i}' : 
            (
                EncoderLayer(
                    n_heads,
                    d_model,
                    d_ff,
                    device = device
                )
            ) for i in range(n_layer)
            })

    def forward(self, x, encoder_mask) :
        # logs(f'input size : {x.size()}')
        for name, layer in self.encoder.items() :
            x, attention_scores = layer(x, encoder_mask)
            # logs(f'{name} output size : {x.size()}')
        return x, attention_scores




###################################### DECODER ######################################

class Decoder(nn.Module) :

    def __init__(self,
                 n_layer: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 device: str = 'cuda'
    ) -> None :

        super(Decoder, self).__init__()

        self.decoder = nn.ModuleDict({
            f'decoder_layer_{i}' :
            (
                DecoderLayer(
                    n_heads,
                    d_model,
                    d_ff,
                    device = device
                )
            ) for i in range(n_layer)
        })


    def forward(self, x, enc_out) :

        for name, layer in self.decoder.items() :
            x = layer(x, enc_out)
            
        return x




###################################### TRANSFORMER_FMT ######################################

class Transformers(nn.Module) :

    def __init__(self,
                 n_layer,
                 n_heads,
                 d_model,
                 d_ff,
                 max_seq_len,
                 vocab_size,
                 device
        ) -> None :

        super(Transformers, self).__init__()

        vocab_size = vocab_size + 2

        self.encoder = Encoder(n_layer, n_heads, d_model, d_ff, device=device)
        self.decoder = Decoder(n_layer, n_heads, d_model, d_ff, device=device)
        self.positonal_embedding = PositionEmbedding(max_seq_len, d_model, device=device)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.logit_layer = nn.Linear(d_model, vocab_size)

        self.max_seq_len = max_seq_len


    def encoder_pass(self, x, encoder_mask) :
        
        x = self.embedding(x)
        x = self.positonal_embedding(x)
        x = self.encoder(x, encoder_mask)

        return x
    

    def decoder_pass(self, enc_output, input_ids) :

        x = self.embedding(input_ids)
        x = self.positonal_embedding(x)
        x = self.decoder(x, enc_output)

        next_token_logits = F.relu(self.logit_layer(x))
        next_token_logits = next_token_logits.reshape(-1, next_token_logits.size(2))

        return F.log_softmax(next_token_logits, dim=1)

    def forward(self, encoder_inp, decoder_inp, encoder_mask) :

        enc_output, attention_scores = self.encoder_pass(encoder_inp, encoder_mask)
        output = self.decoder_pass(enc_output, decoder_inp)

        return attention_scores, output