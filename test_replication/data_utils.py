# -*- coding: utf-8 -*-
"""
Data utilities for Chinese punctuation restoration
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_file(filename):
    """
    Load data file
    Format: each line is "word punctuation"
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


def encode_data(data, tokenizer, punctuation_enc):
    """
    Convert words to BERT tokens and punctuation to encoding.
    Note: words can be composed of multiple tokens.
    
    Args:
        data: list of lines, each line is "word punctuation"
        tokenizer: BERT tokenizer
        punctuation_enc: dictionary mapping punctuation to label id
    
    Returns:
        X: list of token ids
        Y: list of punctuation labels
    """
    X = []
    Y = []
    for line in data:
        word, punc = line.split()
        punc = punc.strip()
        
        tokens = tokenizer.tokenize(word)
        x = tokenizer.convert_tokens_to_ids(tokens)
        y = [punctuation_enc[punc]]
        
        if len(x) > 0:
            # If word is split into multiple tokens, assign punctuation to last token
            if len(x) > 1:
                y = (len(x) - 1) * [0] + y  # 0 = no punctuation
            X += x
            Y += y
    
    return X, Y


def preprocess_data(data, tokenizer, punctuation_enc, seq_len):
    """
    Preprocess data into fixed-length sequences
    
    Args:
        data: list of lines
        tokenizer: BERT tokenizer
        punctuation_enc: punctuation encoding dictionary
        seq_len: sequence length (e.g., 200)
    
    Returns:
        X: numpy array [num_sequences, seq_len]
        Y: numpy array [num_sequences, seq_len]
    """
    X, Y = encode_data(data, tokenizer, punctuation_enc)
    length = len(X)
    X = np.array(X)
    Y = np.array(Y)
    
    # Remove remainder to make sequences fit exactly
    remain = length % seq_len
    if remain > 0:
        X = X[:-remain].reshape((-1, seq_len))
        Y = Y[:-remain].reshape((-1, seq_len))
    else:
        X = X.reshape((-1, seq_len))
        Y = Y.reshape((-1, seq_len))
    
    return X, Y


def create_data_loader(X, y, shuffle, batch_size):
    """
    Create PyTorch DataLoader
    
    Args:
        X: input sequences [num_sequences, seq_len]
        y: label sequences [num_sequences, seq_len]
        shuffle: whether to shuffle data
        batch_size: batch size
    
    Returns:
        DataLoader object
    """
    data_set = TensorDataset(
        torch.from_numpy(X).long(),
        torch.from_numpy(np.array(y)).long()
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_loader

