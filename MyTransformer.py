import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size:int, d_model:int):
        super(TokenEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, tokens: torch.Tensor):
        return self.embedding_layer(tokens.long()) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, max_length_seq:int, d_model:int, dropout_rate:float):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, d_model, 2)* math.log(10000) / d_model)
        pos = torch.arange(0, max_length_seq).reshape(max_length_seq, 1)
        PE = torch.zeros((max_length_seq, d_model))
        PE[:, 0::2] = torch.sin(pos * den)
        PE[:, 1::2] = torch.cos(pos * den)
        PE = PE.unsqueeze(-2)
        self.register_buffer("PE", PE)
        self.dropout_layer = nn.Dropout(dropout_rate)
        
    def forward(self, token_embedding: torch.Tensor):
        return self.dropout_layer(token_embedding + self.PE[:token_embedding.size(0), :])
    
class TransformerModel(nn.Module):
    def __init__(self,
                 src_vocab_size:int,
                 tgt_vocab_size:int,
                 d_model:int,
                 nhead:int,
                 num_encoder_layers:int,
                 num_decoder_layers:int,
                 dim_feedforward:int,
                 dropout:float,
                 max_length_seq=20000):
        super(TransformerModel,self).__init__()
        # Input Preprocessing
        self.input_token_embedding = TokenEmbedding(vocab_size=src_vocab_size,d_model=d_model)
        
        # Output Preprocessing
        self.output_token_embedding = TokenEmbedding(vocab_size=tgt_vocab_size,d_model=d_model)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(max_length_seq=max_length_seq,d_model=d_model,dropout_rate=dropout)
        
        # Transformer Architecture with Encoder & Decoder (available nn.Transformer) 
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        
        # Linear Layer
        self.generator = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                tgt_key_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        
        # Pre-processing
        src_embedding = self.positional_encoding(self.input_token_embedding(src))
        tgt_embedding = self.positional_encoding(self.output_token_embedding(tgt))
        
        output = self.transformer(src_embedding, tgt_embedding, src_mask, tgt_mask, None, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.generator(output)
    
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.input_token_embedding(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.output_token_embedding(tgt)), memory,
                          tgt_mask)
        