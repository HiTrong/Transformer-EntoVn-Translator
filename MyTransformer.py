# import library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

if torch.cuda.is_available():
    print("CUDA is available. PyTorch is using GPU.")
    print("Number of GPUs available: ", torch.cuda.device_count())
    print("GPU name: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. PyTorch is using CPU.")
    

# ---------------------------------------------------------------- Modules ----------------------------------------------------------------
class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size, d_model):
        """
        Token Embedding is used for converting a word / token into a embedding numeric vector space.
        
        :param vocab_size: Number of words / token in vocabulary
        :param d_model: The embedding dimension
        
        Example: With 1000 words in vocabulary and our embedding dimension is 512, the Token Embedding layer will be 1000x512
        """
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding_layer = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        :param x: the word or sequence of words
        :return: the numerical representation of the input
        
        Example:
        Input: (Batch_size, Sequence of words) - (30x100)
        Output: (Batch_size, Sequence of words, d_model) - (30x100x512)
        """
        output = self.embedding_layer(x)
        return output
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_sequence_length, dropout=0):
        """
        Positional Encoding layer for adding positional information to token embeddings.
        
        :param d_model: The embedding dimension.
        :param max_sequence_length: The maximum length of the input sequences.
        :param dropout: Dropout rate.
        """
        super(PositionalEncoding,self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        PE = PE.unsqueeze(0)
        return self.dropout(PE)
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads=8):
        """
        Multi-Head Attention
        :param d_model: the embedding dimension
        :param num_heads: the number of heads, default equals 8
        
        # note: The embedding dimension must be divided by the number of heads
        """
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # query, key value
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)  # the Query metrix
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)  # the Key metrix
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)  # the Value metrix
        
        
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Perform forward pass of the multi-head attention mechanism.

        :param query: Query tensor of shape (batch_size, q_len, d_model)
        :param key: Key tensor of shape (batch_size, k_len, d_model)
        :param value: Value tensor of shape (batch_size, v_len, d_model)
        :param mask: Optional mask tensor of shape (batch_size, 1, 1, k_len)
        
        :return: Output tensor of shape (batch_size, q_len, d_model)

        """
        # Input of size: batch_size x sequence length x embedding dims
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)

        # reshape from (batch_size x seq_len x embed_size) -> (batch_size x seq_len x heads x head)
        # example: from (30x10x512) -> (30x10x8x64)
        key = key.reshape(batch_size, k_len, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, q_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, v_len, self.num_heads, self.head_dim)

        key = self.key(key)  # (30x10x8x64)
        query = self.query(query)  # (30x10x8x64)
        value = self.value(value)  # (30x10x8x64)

        # query shape: batch_size x q_len, heads, head, e.g: (30x10x8x64)
        # key shape: batch_size x v_len, heads, head, e.g: (30x10x8x64)
        # product shape should be: batch_size, heads, q_len, v_len, e.g: (30x8x10x10)
        product = torch.einsum("bqhd,bkhd->bhqk", [query, key])

        # if mask (in decoder)
        if mask is not None:
            mask = mask.to(query.device)
            product = product.masked_fill(mask == 0, float("-1e20")) # -inf for softmax -> 0

        product = product / math.sqrt(self.head_dim)

        scores = F.softmax(product, dim=-1)

        # scores shape: batch_size, heads, q_len, v_len, e.g: (30x8x10x10)
        # value shape: batch_size, v_len, heads, head, e.g: (30x10x8x64)
        # output: batch_size, heads, v_len, head, e.g: (30x10x512)
        output = torch.einsum("nhql,nlhd->nqhd", [scores, value]).reshape(
            batch_size, q_len, self.num_heads * self.head_dim
        )

        output = self.linear_layer(output)  # (30x10x512) -> (30x10x512)
        
        return output
    
def replicate(block, N=6) -> nn.ModuleList:
    """
    Method to replicate the existing block to N set of blocks
    :param block: class inherited from nn.Module, mainly it is the encoder or decoder part of the architecture
    :param N: the number of stack, in the original paper they used 6
    :return: a set of N blocks
    """
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack

class TransformerBlock(nn.Module):

    def __init__(self,
                 d_model=512,
                 num_heads=8,
                 expansion_factor=4,
                 dropout=0.1
                ):
        """
        The Transformer Block used in the encoder and decoder as well

        :param d_model: the embedding dimension
        :param num_heads: the number of heads
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        """
        super(TransformerBlock, self).__init__()

        self.multihead_attention = MultiHeadAttention(d_model,num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, expansion_factor * d_model), # Ex: (512,1024)
            nn.ReLU(),
            nn.Linear(expansion_factor * d_model, d_model), # Ex: (1024,512)
            # The output shape will be not different from input
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # First come to Multi-Head Attention
        attention = self.multihead_attention(query,key,value,mask)

        # Add & Norm
        # Add
        attention_added = attention + value;
        # Norm 
        attention_norm = self.dropout(self.norm(attention_added))

        # Feed & Forward
        attention_ff = self.feed_forward(attention_norm)

        # Add & Norm again!
        # Add
        attention_ff_added = attention_ff + attention_norm
        # Norm
        attention_ff_norm = self.dropout(self.norm(attention_ff_added))

        return attention_ff_norm
        
class Encoder(nn.Module):

    def __init__(self,
                 max_length_seq,
                 vocab_size,
                 d_model=512,
                 num_blocks=6,
                 expansion_factor=4,
                 num_heads=8,
                 dropout=0.1
                ):
        """
        The Encoder part of the Transformer architecture

        :param max_length_seq: the max length of the sequence
        :param vocab_size: the total size of the vocabulary
        :param d_model: the embedding dimension
        :param num_blocks: the number of blocks (encoders), 6 by default
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer in each encoder
        :param num_heads: the number of heads in each encoder
        :param dropout: probability dropout (between 0 and 1)
        """
        super(Encoder, self).__init__()

        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Token Embedding
        self.token_emb = TokenEmbedding(vocab_size,d_model)
        # Positional Encoding
        self.pos_encode = PositionalEncoding(d_model,max_length_seq)

        # Transformer Blocks
        self.transformer_blocks = replicate(TransformerBlock(d_model,num_heads,expansion_factor,dropout),num_blocks)

    def forward(self,x):
        # Input Pre-processing: Token Embedding + Positional Encoding
        pos = self.pos_encode().to(x.device)
        output = self.dropout(pos[:, :x.size(1), :].requires_grad_(False) + self.token_emb(x))

        # Go to Transformer Blocks (Encode)
        for block in self.transformer_blocks:
            output = block(output,output,output)

        return output
                 
class DecoderBlock(nn.Module):

    def __init__(self,
                 d_model=512,
                 num_heads=8,
                 expansion_factor=4,
                 dropout=0.1
                ):
        """
        The DecoderBlock which will consist of the TransformerBlock used in the encoder, plus a decoder multi-head attention
        :param d_model: the embedding dimension
        :param num_heads: the number of heads
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        """
        super(DecoderBlock, self).__init__()

        # Masked Multi-Head Attention
        self.attention = MultiHeadAttention(d_model,num_heads)

        # Normalization in Add & Norm
        self.norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer Block
        self.transformer_block = TransformerBlock(d_model,num_heads,expansion_factor,dropout)

    def forward(self, query, key, x, mask): # Different from Encoder
        # Masked Multi-Head Attention
        decoder_attention = self.attention(x,x,x, mask)

        # Add & Norm
        # Add
        decoder_attention_added = self.dropout(decoder_attention + x)
        # Norm
        decoder_attention_norm = self.dropout(self.norm(decoder_attention_added))

        # Transformer Block
        decoder_attention_output = self.transformer_block(query, key, decoder_attention_norm)

        return decoder_attention_output

class Decoder(nn.Module):

    def __init__(self,
                 target_vocab_size,
                 max_length_seq,
                 d_model=512,
                 num_blocks=6,
                 expansion_factor=4,
                 num_heads=8,
                 dropout=0.1
                ):
        """
        The Decoder part of the Transformer architecture

        :param target_vocab_size: the size of the target
        :param max_length_seq: the length of the sequence, in other words, the length of the words
        :param d_model: the embedding dimension
        :param num_blocks: the number of blocks (encoders), 6 by default
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer in each decoder
        :param num_heads: the number of heads in each decoder
        :param dropout: probability dropout (between 0 and 1)
        """
        super(Decoder, self).__init__()
        
         # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Token Embedding
        self.token_emb = TokenEmbedding(target_vocab_size,d_model)
        # Positional Encoding
        self.pos_encode = PositionalEncoding(d_model,max_length_seq)

        # Decoder Blocks
        self.decoder_blocks = replicate(DecoderBlock(d_model,num_heads,expansion_factor,dropout), num_blocks)

    def forward(self, x, encoder_output, mask):
        # Output Pre-processing: Token Embedding + Positional Encoding
        pos = self.pos_encode().to(x.device)
        output = self.dropout(pos[:, :x.size(1), :].requires_grad_(False) + self.token_emb(x))

        # Go to Transformer Blocks (Encode)
        for block in self.decoder_blocks:
            output = block(encoder_output,encoder_output,output, mask)

        return output
    
class Transformer(nn.Module):

    def __init__(self,
                 d_model,
                 vocab_size,
                 target_vocab_size,
                 max_length_seq,
                 num_blocks=6,
                 expansion_factor=4,
                 num_heads=8,
                 dropout=0.1
                ):
        super(Transformer, self).__init__()

        self.target_vocab_size = target_vocab_size

        self.encoder = Encoder(max_length_seq=max_length_seq,
                              vocab_size=vocab_size,
                               d_model=d_model,
                               num_blocks=num_blocks,
                               expansion_factor=expansion_factor,
                               num_heads=num_heads,
                               dropout=dropout)

        self.decoder = Decoder(target_vocab_size=target_vocab_size,
                              max_length_seq=max_length_seq,
                              d_model=d_model,
                              num_blocks=num_blocks,
                              expansion_factor=expansion_factor,
                              num_heads=num_heads,
                              dropout=dropout)

        self.linear_layer = nn.Linear(d_model, target_vocab_size)

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def forward(self, source, target):
        trg_mask = self.make_trg_mask(target)
        enc_out = self.encoder(source)
        outputs = self.decoder(target, enc_out, trg_mask)
        output = F.softmax(self.linear_layer(outputs), dim=-1)
        return output