import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers_si.model_blocks.transformer_layer import EncoderLayer, DecoderLayer

from utils.logging import logs

class PositionEmbedding(nn.Module) :

    def __init__(self,
        max_seq_len: int = 64, 
        d_model: int = 512,
    ) :

        super(PositionEmbedding, self).__init__()

        self.embedding = torch.zeros(max_seq_len, d_model)
        
        for i in range(max_seq_len) :
            self.embedding[i, 0::2] = torch.sin((i/1000**(2*torch.arange(512)[::2]/512)))
            self.embedding[i, 1::2] = torch.cos((i/1000**(2*torch.arange(512)[1::2]/512)))

        # self.embedding = torch.repeat_interleave(self.embedding.unsqueeze(0), batch_size, 0)

    def forward(self, x) :
        self.embedding = torch.repeat_interleave(self.embedding.unsqueeze(0), x.size(0), 0)
        return x + self.embedding[:, :x.size(1), :]
    

class Encoder(nn.Module) :

    def __init__(self,
                 n_layer: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_ff: int = 2048
    ) :
        super(Encoder, self).__init__()
        
        self.encoder = nn.ModuleDict({
            f'encoder_layer_{i}' : 
            (
                EncoderLayer(
                    n_heads,
                    d_model,
                    d_ff
                )
            ) for i in range(n_layer)
            })

    def forward(self, x) :
        # logs(f'input size : {x.size()}')
        for name, layer in self.encoder.items() :
            x, attention_scores = layer(x)
            # logs(f'{name} output size : {x.size()}')
        return x, attention_scores
    
class Decoder(nn.Module) :

    def __init__(self,
                 n_layer: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_ff: int = 2048
    ) -> None :

        super(Decoder, self).__init__()

        self.decoder = nn.ModuleDict({
            f'decoder_layer_{i}' :
            (
                DecoderLayer(
                    n_heads,
                    d_model,
                    d_ff
                )
            ) for i in range(n_layer)
        })


    def forward(self, x, enc_out) :

        for name, layer in self.decoder.items() :
            x = layer(x, enc_out)
            
        return x


class Transformers(nn.Module) :

    def __init__(self,
                 n_layer,
                 n_heads,
                 d_model,
                 d_ff,
                 max_seq_len,
                 vocab_size,
        ) -> None :

        super(Transformers, self).__init__()

        vocab_size += 3

        self.encoder = Encoder(n_layer, n_heads, d_model, d_ff)
        self.decoder = Decoder(n_layer, n_heads, d_model, d_ff)
        self.positonal_embedding = PositionEmbedding(max_seq_len, d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.logit_layer = nn.Linear(d_model, vocab_size)

        self.max_seq_len = max_seq_len
        self.pad_token_id = 12
        self.bos_token_id = 0
        self.eos_token_id = 13

    
    def preprocess(self, inputs_token_ids) :
        
        input_token_ids = [
            torch.cat(
                (torch.tensor([self.bos_token_id]), inp),
            ) for inp in inputs_token_ids
        ]

        input_token_ids = pad_sequence(input_token_ids, batch_first=True, padding_value = self.pad_token_id)

        input_token_ids = torch.cat(
            (input_token_ids, torch.tensor([[self.eos_token_id] for _ in range(input_token_ids.size(0))])),
            dim = 1
        )

        return input_token_ids

    def encode(self, x) :
        logs('encoding_now ------------------------------------')
        x = self.embedding(x)
        x = self.positonal_embedding(x)
        x = self.encoder(x)

        return x

    def generate(self, enc_output, input_ids) :

        logs('generating_now ------------------------------------')

        x = self.embedding(input_ids)
        x = self.positonal_embedding(x)

        sentence_length = input_ids.size(1)

        while sentence_length < self.max_seq_len :

            x = self.embedding(input_ids)
            x = self.positonal_embedding(x)
            x = self.decoder(x, enc_output)
            next_token_logits = x[:, -1, :]
            next_token_logits = self.logit_layer(next_token_logits)
            
            # logs(f'next_token_logits size: {next_token_logits.size()}')
            next_token_logits = F.softmax(next_token_logits, dim=1)
            next_token_indices = torch.argmax(next_token_logits, dim = 1)
            # logs(f'next_token_logits size: {next_token_indices.size()}')

            input_ids = torch.cat(
                (
                    input_ids, 
                    next_token_indices.unsqueeze(1)
                ), 
                dim = 1)

            sentence_length += 1

            return input_ids

    def forward(self, x: list) :
        
        assert type(x) == list, "x must be a list of tensors"

        x = self.preprocess(x)
        # logs(f'input : {x}')
        # logs(f'input_shape : {x.size()}')
        enc_output, attention_scores = self.encode(x)
        input_ids = self.generate(enc_output, x)

        return input_ids, attention_scores
