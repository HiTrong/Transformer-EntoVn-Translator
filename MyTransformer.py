# import library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# ======================================== Module ========================================
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def Masking(encoder_batch, decoder_batch, max_length_seq):
    NEG_INFTY = -1e9
    num_sentences = len(encoder_batch)
    look_ahead_mask = torch.full([max_length_seq, max_length_seq] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_length_seq, max_length_seq] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_length_seq, max_length_seq] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_length_seq, max_length_seq] , False)

    for idx in range(num_sentences):
      encoder_sentence_length, decoder_sentence_length = len(encoder_batch[idx]), len(decoder_batch[idx])
      encoder_chars_to_padding_mask = np.arange(encoder_sentence_length + 1, max_length_seq)
      decoder_chars_to_padding_mask = np.arange(decoder_sentence_length + 1, max_length_seq)
      encoder_padding_mask[idx, :, encoder_chars_to_padding_mask] = True
      encoder_padding_mask[idx, encoder_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, decoder_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, decoder_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, encoder_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, decoder_chars_to_padding_mask, :] = True
      

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0).to(get_device())
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0).to(get_device())
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0).to(get_device())
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask
     

class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size, d_model):
        """
        Token Embedding is used for converting a word / token into a embedding numeric vector space.
        
        :param vocab_size: Number of words / token in vocabulary
        :param d_model: The embedding dimension
        
        Example: With 1000 words in vocabulary and our embedding dimension is 512, the Token Embedding layer will be 1000x512
        """
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        :param x: the word or sequence of words
        :return: the numerical representation of the input
        
        Example:
        Input: (Batch_size, Sequence of words) - (30x100)
        Output: (Batch_size, Sequence of words, d_model) - (30x100x512)
        """
        x = self.embedding_layer(x)
        return x.to(get_device())

# Or just Simple
# token_embedding = nn.Embedding(vocab_size, d_model)

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_sequence_length, dropout=0.1):
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
        PE = self.get()
        self.register_buffer('PE', PE)
        
    def get(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        PE = PE.unsqueeze(0)
        return PE


    def forward(self):
        return self.dropout(self.PE)
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads=8, cross=False):
        """
        Multi-Head Attention
        :param d_model: the embedding dimension
        :param num_heads: the number of heads, default equals 8
        :param cross: True for Multi-Head Cross Attention, False for Multi-Head Attention only
        
        # note: The embedding dimension must be divided by the number of heads
        """
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.cross = cross

        # query, key value layer
        if self.cross: # Multi-Head Cross Attention
            self.kv_layer = nn.Linear(d_model , 2 * d_model)
            self.q_layer = nn.Linear(d_model , d_model)
        else:
            self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        
        
        # method 1: old, cost alot
        # self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.value = nn.Linear(self.head_dim, self.head_dim, bias=False) 

        # method 2: the fewer linear layers the better the cost
        
        
        # Linear Layer in Multi-Head Attention
        self.linear_layer = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            scaled = scaled.permute(1, 0, 2, 3) + mask
            scaled = scaled.permute(1, 0, 2, 3)
        attention = F.softmax(scaled, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention
    
    def forward(self, x, mask=None):
        """
        Perform forward pass of the multi-head attention mechanism.

        :param x: if cross is True then x is a dictionary including  'encoder_output' and 'w'.
        :param mask: Optional mask tensor
        
        :return: Output tensor of shape (batch_size, length_seq, d_model)

        """

        # For MultiHead Cross Attention
        if self.cross:
            encoder_output = x['encoder_output']
            w = x['w']
            batch_size, length_seq, d_model = w.size()
            kv = self.kv_layer(w)
            q = self.q_layer(encoder_output)
            kv = kv.reshape(batch_size, length_seq, self.num_heads, 2 * self.head_dim)
            q = q.reshape(batch_size, length_seq, self.num_heads, self.head_dim)
            kv = kv.permute(0, 2, 1, 3)
            q = q.permute(0, 2, 1, 3)
            k, v = kv.chunk(2, dim=-1)
            values, attention = self.scaled_dot_product(q, k, v, mask) # mask is not required in Cross Attention
            values = values.permute(0, 2, 1, 3).reshape(batch_size, length_seq, self.num_heads * self.head_dim)
            out = self.linear_layer(values)
            return out

        # For MultiHead Attention
        batch_size, length_seq, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, length_seq, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = self.scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, length_seq, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out
    
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y  + self.beta
        return out

# Or using nn.LayerNorm(d_model)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# feed_forward = nn.Sequential(
#     nn.Linear(d_model, expansion_factor * d_model),  # e.g: 512x(4*512) -> (512, 2048)
#     nn.ReLU(),  # ReLU activation function
#     nn.Linear(d_model * expansion_factor, d_model),  # e.g: 4*512)x512 -> (2048, 512)
# )

def replicate(block, N=6) -> nn.ModuleList:
    """
    Method to replicate the existing block to N set of blocks
    :param block: class inherited from nn.Module, mainly it is the encoder or decoder part of the architecture
    :param N: the number of stack, in the original paper they used 6
    :return: a set of N blocks
    """
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack

class Preprocessing(nn.Module):

    def __init__(self, max_length_seq, d_model, language_to_index, start_token, end_token, pad_token, dropout=0.1):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.language_to_index = language_to_index
        self.max_length_seq = max_length_seq
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

        # Layer
        self.token_embedding = TokenEmbedding(self.vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length_seq, dropout)
        self.dropout = nn.Dropout(dropout)

    
    
    def batch_tokens(self, batch, start_token:bool, end_token:bool):
        def tokenize(sentence, start_token:bool, end_token:bool):
            encode_char = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                encode_char.insert(0, self.language_to_index[self.start_token])
            if end_token:
                encode_char.append(self.language_to_index[self.end_token])
            for _ in range(len(encode_char), self.max_length_seq):
                encode_char.append(self.language_to_index[self.pad_token])
            return torch.tensor(encode_char)
        
        tokens = []
        for i in range(len(batch)):
            tokens.append(tokenize(batch[i], start_token, end_token))
        tokens = torch.stack(tokens)
        return tokens

    def forward(self, x, start_token:bool, end_token:bool): 
        x = self.batch_tokens(x, start_token, end_token)
        x = self.token_embedding(x.to(get_device()))
        pos = self.positional_encoding().to(get_device())
        x = self.dropout(x + pos)
        return x
    
class TransformerBlock(nn.Module):

    def __init__(self,
                 d_model=512,
                 num_heads=8,
                 ff_hidden=300,
                 dropout=0.1,
                 options='encoder'
                ):
        """
        The Transformer Block used in the encoder and decoder as well

        :param d_model: the embedding dimension
        :param num_heads: the number of heads
        :param ff_hidden: The output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        :param options: The choice between 'encoder' and 'decoder'
        """
        super(TransformerBlock, self).__init__()
    
        self.options = options
        
        # For both 2 options: encoder and decoder
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm_for_attention = LayerNormalization(parameters_shape=[d_model])
        self.dropout_attention = nn.Dropout(dropout)

        
        
        # For decoder
        if self.options=='decoder':
            self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, cross=True)
            self.norm_for_cross_attention = LayerNormalization(parameters_shape=[d_model])
            self.dropout2 = nn.Dropout(dropout)
        elif self.options!='encoder':
            raise Exception(f"Unknown option {options}")

        # For both 2 options: encoder and decoder
        self.ff = PositionwiseFeedForward(d_model=d_model, hidden=ff_hidden, drop_prob=dropout)
        self.norm_for_ff = LayerNormalization(parameters_shape=[d_model])
        self.dropout_for_ff = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # For decoder
        if self.options == 'decoder':
            encoder_output = x['encoder_output']
            w = x['w']
            w_residual = w.clone()
            w = self.attention(w,mask['self_attention_mask'])
            w = self.dropout_attention(w)
            w = self.norm_for_attention(w + w_residual)

            w_residual = w.clone()
            w = self.cross_attention({'encoder_output':encoder_output,'w':w},mask['cross_attention_mask'])
            w = self.dropout2(w)
            w = self.norm_for_cross_attention(w + w_residual)

            w_residual = w.clone()
            w = self.ff(w)
            w = self.dropout_for_ff(w)
            w = self.norm_for_ff(w + w_residual)
            return w
        else:
        # For encoder
            x_residual = x.clone()
            x = self.attention(x, mask)
            x = self.dropout_attention(x)
            x = self.norm_for_attention(x + x_residual)

            x_residual = x.clone()
            x = self.ff(x)
            x = self.dropout_for_ff(x)
            x = self.norm_for_ff(x + x_residual)
            return x
        
class Encoder(nn.Module):

    def __init__(self,
                 d_model,
                 ff_hidden,
                 num_heads,
                 dropout,
                 num_blocks,
                 max_length_seq,
                 language_to_index,
                 start_token, 
                 end_token, 
                 pad_token
                ):
        """
        The Encoder part of the Transformer architecture
        """
        super().__init__()

        # Layer
        self.input_preprocessing = Preprocessing(max_length_seq, d_model, language_to_index, start_token, end_token, pad_token, dropout)
        
        # Transformer Blocks
        self.transformer_blocks = replicate(TransformerBlock(d_model, num_heads, ff_hidden, dropout, options="encoder"),num_blocks)

    def forward(self, x, self_attention_mask, start_token:bool, end_token:bool):
        # Input Pre-processing: Token Embedding + Positional Encoding
        out = self.input_preprocessing(x, start_token, end_token)

        # Go to Transformer Blocks (Encode)
        for block in self.transformer_blocks:
            out = block(out, self_attention_mask)

        return out
    
class Decoder(nn.Module):

    def __init__(self,
                 d_model,
                 ff_hidden,
                 num_heads,
                 dropout,
                 num_blocks,
                 max_length_seq,
                 language_to_index,
                 start_token, 
                 end_token, 
                 pad_token
                ):
        """
        The Decoder part of the Transformer architecture

        """
        super().__init__()
        
         # Layer
        self.output_preprocessing = Preprocessing(max_length_seq, d_model, language_to_index, start_token, end_token, pad_token, dropout)
        
        # Transformer Blocks
        self.transformer_blocks = replicate(TransformerBlock(d_model, num_heads, ff_hidden, dropout, options="decoder"),num_blocks)

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token:bool, end_token:bool): 
        # x is output, y is output from encoder
        # Output Pre-processing: Token Embedding + Positional Encoding
        x = self.output_preprocessing(x, start_token, end_token)

        # Go to Transformer Blocks (Decode)
        encode_decode = {'encoder_output': y,'w':x}
        mask = {'self_attention_mask': self_attention_mask,'cross_attention_mask': cross_attention_mask}
        for block in self.transformer_blocks:
            encode_decode['w'] = x
            x = block(encode_decode, mask)
        return x
    
class Transformer(nn.Module):

    def __init__(self,
                 d_model,
                 ff_hidden,
                 num_heads,
                 dropout,
                 num_blocks,
                 max_length_seq,
                 language_to_index,
                 target_language_to_index,
                 start_token, 
                 end_token, 
                 pad_token
                ):
        super().__init__()

        # Device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Encoder
        self.encoder = Encoder(
            d_model=d_model,
            ff_hidden=ff_hidden,
            num_heads=num_heads,
            dropout=dropout,
            num_blocks=num_blocks,
            max_length_seq=max_length_seq,
            language_to_index=language_to_index,
            start_token=start_token,
            end_token=end_token,
            pad_token=pad_token
        )

        # Decoder
        self.decoder = Decoder(
            d_model=d_model,
            ff_hidden=ff_hidden,
            num_heads=num_heads,
            dropout=dropout,
            num_blocks=num_blocks,
            max_length_seq=max_length_seq,
            language_to_index=target_language_to_index,
            start_token=start_token,
            end_token=end_token,
            pad_token=pad_token
        )

        # Linear Layer
        self.linear = nn.Linear(d_model, len(target_language_to_index))

        # Softmax
        

    def forward(self,
                x,
                y,
                encoder_self_attention_mask=None,
                decoder_self_attention_mask=None,
                decoder_cross_attention_mask=None,
                encoder_start_token=False,
                encoder_end_token=False,
                decoder_start_token=False,
                decoder_end_token=False):
        encoder_output = self.encoder(x, encoder_self_attention_mask, encoder_start_token, encoder_end_token)
        out = self.decoder(y, encoder_output, decoder_self_attention_mask, decoder_cross_attention_mask, decoder_start_token, decoder_end_token)
        out = self.linear(out)
        return out