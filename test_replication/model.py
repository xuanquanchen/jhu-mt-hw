# -*- coding: utf-8 -*-
"""
BertChineseEmbSlimCNNlstmBert - Best Model for Chinese Punctuation Restoration
Combines BERT embeddings, CNN layers, and LSTM for word-level and character-level features
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from torch import nn
from transformers import BertModel


class BertChineseEmbSlimCNNlstmBert(nn.Module):
    """
    Best model for Chinese punctuation restoration.
    Architecture:
    - BERT embeddings + CNN layers (word-level features)
    - Second BERT (character-level features)
    - LSTM layer
    - Final classification layer
    """
    
    def __init__(self, segment_size, output_size, dropout, vocab_size=None):
        super(BertChineseEmbSlimCNNlstmBert, self).__init__()
        
        # Two BERT models
        # bert: for CNN processing (uses embeddings)
        # bert_2: for character-level features
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert_2 = BertModel.from_pretrained('bert-base-chinese')
        
        self.bert_size = 768
        
        # CNN layers for word-level features
        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, self.bert_size)
        cnn_filter_num = self.bert_size
        cnn_layer_num = 5
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num
        
        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i): nn.Conv2d(
                    1,
                    self.cnn_filter_num,
                    self.cnn_kernel_size,
                    padding=((cnn_kernel_size[0] - 1) // 2, 0)
                ),
                'conv_v_{}'.format(i): nn.Conv2d(
                    1,
                    self.cnn_filter_num,
                    self.cnn_kernel_size,
                    padding=((cnn_kernel_size[0] - 1) // 2, 0)
                )
            })
            self.conv.append(module_tmp)
        
        # LSTM layer
        self.lstm_cnn = nn.LSTM(
            self.bert_size,
            self.bert_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        
        # Final classification layer
        self.dropout = nn.Dropout(dropout)
        # LSTM output: bert_size * 2 (bidirectional) + bert_2 output: bert_size
        self.fc_size = self.bert_size + self.bert_size * 2
        self.fc = nn.Linear(self.fc_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: input token ids [batch_size, seq_len]
        Returns:
            output: punctuation predictions [batch_size * seq_len, num_classes]
        """
        # Get character-level features from second BERT
        emb2 = self.bert_2(x)[0]  # [batch_size, seq_len, 768]
        
        # Get embeddings from first BERT for CNN processing
        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None
        
        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)
        
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)
        
        # Get BERT embeddings
        embedding_output = self.bert.embeddings(
            input_ids=input_ids, 
            position_ids=position_ids, 
            token_type_ids=token_type_ids, 
            inputs_embeds=inputs_embeds
        )
        
        # CNN layers for word-level features (with skip connections)
        skip_connection = embedding_output.unsqueeze(1)  # [batch, 1, seq_len, 768]
        cnn_out = embedding_output.unsqueeze(1)
        
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)  # Gated activation
            
            # Concatenate filters to maintain size
            cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
            cnn_out = cnn_out.unsqueeze(1)
            
            # Skip connection
            cnn_out = skip_connection + cnn_out
            skip_connection = cnn_out
        
        # Remove channel dimension
        output1 = cnn_out.squeeze(1)  # [batch, seq_len, 768]
        
        # LSTM processing
        output1, _ = self.lstm_cnn(output1)  # [batch, seq_len, 768*2]
        
        # Combine CNN+LSTM features with BERT character-level features
        x = torch.cat([output1, emb2], dim=-1)  # [batch, seq_len, 768*2 + 768]
        
        # Reshape for classification
        x = x.view(-1, x.shape[2])  # [batch*seq_len, 768*3]
        
        # Final classification
        x = self.fc(self.dropout(x))  # [batch*seq_len, num_classes]
        
        return x

