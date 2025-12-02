# -*- coding: utf-8 -*-
"""
BertPunc - Baseline Model for Chinese Punctuation Restoration
Original model from BertPunc repository, adapted for Chinese
Uses BERT outputs flattened and passed through a linear layer
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from torch import nn
from transformers import BertModel


class BertPunc(nn.Module):
    """
    Baseline model for Chinese punctuation restoration.
    Architecture:
    - BERT model (bert-base-chinese)
    - Flatten BERT outputs
    - Batch normalization
    - Linear classification layer
    """
    
    def __init__(self, segment_size, output_size, dropout, vocab_size=None):
        super(BertPunc, self).__init__()
        
        # Use bert-base-chinese for Chinese text
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert_size = 768  # BERT base hidden size
        self.segment_size = segment_size
        
        # Batch normalization for flattened BERT outputs (original BertPunc approach)
        self.bn = nn.BatchNorm1d(segment_size * self.bert_size)
        
        # Final classification layer for sequence-level prediction (original)
        self.fc = nn.Linear(segment_size * self.bert_size, output_size)
        
        # Per-token classification layer (for per-token predictions)
        self.per_token_fc = nn.Linear(self.bert_size, output_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input token ids [batch_size, seq_len]
        Returns:
            output: punctuation predictions [batch_size * seq_len, num_classes]
        """
        # Get BERT outputs
        # bert_outputs[0] is the sequence of hidden states [batch_size, seq_len, 768]
        bert_outputs = self.bert(x)
        x = bert_outputs[0]  # [batch_size, seq_len, 768]
        
        batch_size, seq_len, hidden_size = x.shape
        
        # For per-token predictions, apply linear layer to each token's hidden state
        # Reshape to [batch_size * seq_len, 768]
        x_per_token = x.contiguous().view(-1, hidden_size)  # [batch_size * seq_len, 768]
        
        # Apply per-token classification
        output = self.per_token_fc(self.dropout(x_per_token))  # [batch_size * seq_len, output_size]
        
        return output

