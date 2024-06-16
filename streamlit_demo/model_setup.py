# Necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re,string # For Regular Expressions, string handle
from typing import Iterable, List # For building vocab, yield helper
import math
import time
import random
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Dataset hugging face
from datasets import load_dataset

# Natural language Processing & Initializing Vocabulary
from underthesea import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# Building Transformer and Training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence # take the max size and add padding_token to the smaller

# ====================================================== class & Function ======================================================
def vi_tokenize(sentence):
    return word_tokenize(sentence) # word_tokenize from undersea supporting vietnamese

def yield_token_helper(iterator : Iterable, language: str) -> List[str]:
    for num_iter, sample_iter in iterator:
        yield tokenizer[language](sample_iter[language])
        
def vocab_building(df):
    vocab = {}
    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        iterator = df.iterrows()
        vocab[language] = build_vocab_from_iterator(
            iterator=yield_token_helper(iterator, language),
            min_freq=1,
            specials=[UNKNOWN_TOKEN, PADDING_TOKEN, START_TOKEN, END_TOKEN], # List special symbols
            special_first=True
        )

    for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab[language].set_default_index(UNKNOWN_IDX)
    return vocab

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
        
class SupportTransformer:
    
    def __init__(self,
                 device,
                 src_language:str,
                 tgt_language:str,
                 start_idx:int,
                 end_idx:int,
                 pad_idx:int,
                 unk_idx:int,
                 tokenizer,
                 vocabulary):
        self.device = device
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
    
    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones((size, size), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == self.pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == self.pad_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    def preprocessing_sentece(self, sentence:str, options=True): # True for src, False for tgt
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        sentence = sentence.lower()
        sentence = sentence.strip()
        sentence = re.sub('\s+', ' ', sentence)
        
        
        lang = self.src_language if options else self.tgt_language
        # Tokenizer
        tokens = self.tokenizer[lang](sentence.rstrip("\n"))
        # vocabulary: text -> number
        tokens_idx = self.vocabulary[lang](tokens)
        # Add start_token, end_token and append
        return torch.cat((torch.tensor([self.start_idx]),
                                      torch.tensor(tokens_idx),
                                      torch.tensor([self.end_idx])))
        
    
    def get_batch(self, df):
        return list(zip(df[self.src_language], df[self.tgt_language]))
    
    def preprocessing_batch(self,batch):
        src_out, tgt_out = [], []
        
        for src_data, tgt_data in batch:
            src_out.append(self.preprocessing_sentece(src_data,options=True))
            tgt_out.append(self.preprocessing_sentece(tgt_data,options=False))
            
        src_batch, tgt_batch = pad_sequence(src_out, padding_value=self.pad_idx), pad_sequence(tgt_out, padding_value=self.pad_idx)
    
        return src_batch, tgt_batch
    
    def evaluate(self, model, loss_func, df_valid, batch_size=30, accumulation_steps=5):
        model.eval()
        valid_loss = 0 
        valid_batch = self.get_batch(df_valid)
        valid_dataloader = DataLoader(valid_batch, batch_size=batch_size, collate_fn=self.preprocessing_batch)
        for index, (src, tgt) in enumerate(valid_dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:-1, :] # Without the last word
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)
            
            # predictions
            predictions = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            tgt_out = tgt[1:, :]
            # Loss
            loss = loss_func(predictions.reshape(-1, predictions.shape[-1]), tgt_out.reshape(-1))
            loss = loss / accumulation_steps
            valid_loss += loss.item()
            
        return valid_loss / len(valid_dataloader)
            
        
    
    def train(self, model, optimizer, loss_func, df_train, batch_size = 30, accumulation_steps = 5):
        model.train()
        
        train_loss = 0
        train_batch = self.get_batch(df_train)
        train_dataloader = DataLoader(train_batch, batch_size=batch_size, collate_fn=self.preprocessing_batch)
        
        # Reset grad
        optimizer.zero_grad()
        for index, (src, tgt) in enumerate(train_dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:-1, :] # Without the last word
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)
            
            # predictions
            predictions = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            tgt_out = tgt[1:, :]
            # Loss
            loss = loss_func(predictions.reshape(-1, predictions.shape[-1]), tgt_out.reshape(-1))
            loss = loss / accumulation_steps
            loss.backward()
            
            if (index+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad() # Reset gradients tensor 
            train_loss += loss.item()
            
        return train_loss / len(train_dataloader)
    
    def generate(self, model, src_sentence):
        start_symbol = self.start_idx
        src = self.preprocessing_sentece(src_sentence, True).view(-1, 1)
        max_len = src.shape[0]
        src_mask = (torch.zeros(max_len, max_len)).type(torch.bool)
        
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory.to(self.device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1) # Greedy
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.end_idx:
                break

        return " ".join(
            self.vocabulary[self.tgt_language].lookup_tokens(list(ys.cpu().numpy()))
        ).replace(
            self.vocabulary[self.tgt_language].lookup_token(self.start_idx), ""
        ).replace(
            self.vocabulary[self.tgt_language].lookup_token(self.end_idx), ""
        ).strip()
    
    def completely_generate(self, model, src_sentence):
        start_symbol = self.start_idx
        src = self.preprocessing_sentece(src_sentence, True)
        
        src = src.view(-1, 1)
        max_len = src.shape[0]
        src_mask = (torch.zeros(max_len, max_len)).type(torch.bool)
        
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory.to(self.device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1) # Greedy
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.end_idx:
                break

        # Find unkown words in tgt generate
        tgt_unk_indexes = torch.nonzero(ys == self.unk_idx).squeeze().tolist()
        if isinstance(tgt_unk_indexes, int):
            tgt_unk_indexes = [tgt_unk_indexes]


        output_list = self.vocabulary[self.tgt_language].lookup_tokens(list(ys.cpu().numpy()))
        src_tokens = self.tokenizer[self.src_language](src_sentence.translate(str.maketrans('', '', string.punctuation)))
        for index in tgt_unk_indexes:
            try:
                if index[0] - 1 < len(src_tokens):
                    output_list[index[0]] = src_tokens[index[0] - 1]
            except:
                if index - 1 < len(src_tokens):
                    output_list[index] = src_tokens[index - 1]

        
        return " ".join(
            output_list
        ).replace(
            self.vocabulary[self.tgt_language].lookup_token(self.start_idx), ""
        ).replace(
            self.vocabulary[self.tgt_language].lookup_token(self.end_idx), ""
        ).strip()
    
    def training(self, model, optimizer, loss_func, earlystopping, df_train, df_valid, epochs=5, batch_size = 30, accumulation_steps = 5, custom_test=None):
        history = {'train_loss': [], 'valid_loss': []}
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}: " + ("-" * 80))
            start = time.time()
            train_loss = self.train(model, optimizer, loss_func, df_train, batch_size, accumulation_steps)
            history['train_loss'].append(train_loss)
            valid_loss = self.evaluate(model, loss_func, df_valid, batch_size)
            history['valid_loss'].append(valid_loss)
            print(f"- Train loss: {train_loss:.3f} - Valid loss: {valid_loss:.3f} - Time training: {(time.time() - start):.3f}")
            
            # custom_test
            if custom_test != None:
                print(f"Input '{self.src_language}': {custom_test}")
                print(f"Output '{self.tgt_language}' generate: {self.generate(model, custom_test)}",end="\n\n")
            else:
                print("\n\n")
                
            # EarylyStopping
            earlystopping(train_loss, valid_loss)
            if earlystopping.early_stop:
                print("Early Stopping active!")
                break
        return history
    
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# ====================================================== load ======================================================
def get_supporter(filepath:str):     
    with open(filepath, 'rb') as f:
        support = pickle.load(f)
        support.device = get_device()
        return support
    
def get_model(filepath:str):
    src_vocab_size = 72982
    tgt_vocab_size = 71795
    d_model = 512
    nhead = 8 # d_model must be divisible by nhead
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 512
    dropout = 0.1
    
    model = TransformerModel(src_vocab_size=src_vocab_size,
                         tgt_vocab_size=tgt_vocab_size,
                         d_model=d_model,
                         nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         num_decoder_layers=num_decoder_layers,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout)
    
    model.load_state_dict(torch.load(filepath))
    return model.to(get_device())
