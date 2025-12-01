# -*- coding: utf-8 -*-
"""
BertChineseEmbSlimCNNlstmBert with Adapters - Improved Model
Freezes BERT parameters and uses adapters for fine-tuning
Also includes other improvements:
- Adapter layers for efficient fine-tuning
- Better feature fusion with attention
- Focal Loss support for class imbalance
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
from torch import nn
from transformers import BertModel


class AdapterLayer(nn.Module):
    """
    Adapter layer for efficient fine-tuning.
    Architecture: down_projection -> activation -> up_projection -> residual
    """
    def __init__(self, hidden_size, adapter_size=None, dropout=0.1):
        super(AdapterLayer, self).__init__()
        if adapter_size is None:
            adapter_size = hidden_size // 2  # Default: half of hidden size
        
        self.adapter_size = adapter_size
        self.hidden_size = hidden_size
        
        # Down projection
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        # Activation
        self.activation = nn.GELU()
        # Up projection
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small values for stable training
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # Adapter forward pass
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        # Residual connection
        return residual + x


class AttentionFusion(nn.Module):
    """
    Attention-based feature fusion module.
    Fuses multiple feature representations using attention mechanism.
    """
    def __init__(self, feature_dims, hidden_dim=None):
        super(AttentionFusion, self).__init__()
        if hidden_dim is None:
            hidden_dim = sum(feature_dims) // len(feature_dims)
        
        self.num_features = len(feature_dims)
        self.feature_dims = feature_dims
        
        # Attention weights for each feature
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            ) for dim in feature_dims
        ])
        
        # Output projection
        self.output_proj = nn.Linear(sum(feature_dims), hidden_dim)
    
    def forward(self, features):
        """
        Args:
            features: list of tensors, each [batch_size, seq_len, feature_dim]
        Returns:
            fused: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = features[0].shape[:2]
        
        # Compute attention scores for each feature
        attention_scores = []
        for i, feat in enumerate(features):
            # [batch_size, seq_len, 1]
            score = self.attention[i](feat)
            attention_scores.append(score)
        
        # Stack and normalize
        attention_scores = torch.cat(attention_scores, dim=-1)  # [batch, seq_len, num_features]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch, seq_len, num_features]
        
        # Weighted concatenation
        weighted_features = []
        for i, feat in enumerate(features):
            weight = attention_weights[:, :, i:i+1]  # [batch, seq_len, 1]
            weighted_features.append(feat * weight)
        
        # Concatenate all features
        fused = torch.cat(weighted_features, dim=-1)  # [batch, seq_len, sum(feature_dims)]
        
        # Project to hidden dimension
        fused = self.output_proj(fused)  # [batch, seq_len, hidden_dim]
        
        return fused


class BertChineseEmbSlimCNNlstmBertAdapter(nn.Module):
    """
    Improved model with adapters and better feature fusion.
    Architecture:
    - Frozen BERT embeddings + Adapters
    - CNN layers (word-level features)
    - Second BERT with Adapters (character-level features)
    - LSTM layer
    - Attention-based feature fusion
    - Final classification layer
    """
    
    def __init__(self, segment_size, output_size, dropout, vocab_size=None, 
                 use_adapter=True, adapter_size=None, use_attention_fusion=True):
        super(BertChineseEmbSlimCNNlstmBertAdapter, self).__init__()
        
        self.use_adapter = use_adapter
        self.use_attention_fusion = use_attention_fusion
        self.bert_size = 768
        
        # Two BERT models
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert_2 = BertModel.from_pretrained('bert-base-chinese')
        
        # Freeze BERT parameters
        if use_adapter:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert_2.parameters():
                param.requires_grad = False
            print("✓ BERT parameters frozen")
        
        # Add adapters to BERT outputs if enabled
        if use_adapter:
            if adapter_size is None:
                adapter_size = self.bert_size // 2
            
            # Add adapters after BERT outputs (simpler and still effective)
            self.bert_adapter = AdapterLayer(self.bert_size, adapter_size, dropout)
            self.bert_2_adapter = AdapterLayer(self.bert_size, adapter_size, dropout)
            print(f"✓ Added adapters to BERT outputs (adapter_size={adapter_size})")
        
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
        
        # Feature fusion
        if use_attention_fusion:
            # Use attention to fuse: CNN+LSTM features (bert_size*2) + BERT2 features (bert_size)
            self.fusion = AttentionFusion(
                feature_dims=[self.bert_size * 2, self.bert_size],
                hidden_dim=self.bert_size * 2
            )
            self.fc_size = self.bert_size * 2
        else:
            # Simple concatenation
            self.fc_size = self.bert_size + self.bert_size * 2
        
        # Final classification layer
        self.dropout = nn.Dropout(dropout)
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
        if self.use_adapter:
            # Apply adapter to BERT output
            emb2 = self.bert_2_adapter(emb2)
        
        # Get embeddings from first BERT for CNN processing
        input_ids = x
        attention_mask = torch.ones(input_ids.size(), device=input_ids.device)
        token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device)
        
        # Get BERT embeddings (we use embeddings directly for CNN processing)
        # Note: For CNN processing, we use embeddings, not full BERT output
        # So we don't apply adapter here - adapter is applied to bert_2 output
        embedding_output = self.bert.embeddings(
            input_ids=input_ids, 
            position_ids=None, 
            token_type_ids=token_type_ids, 
            inputs_embeds=None
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
        
        # Feature fusion
        if self.use_attention_fusion:
            # Use attention-based fusion
            x = self.fusion([output1, emb2])  # [batch, seq_len, bert_size*2]
        else:
            # Simple concatenation
            x = torch.cat([output1, emb2], dim=-1)  # [batch, seq_len, 768*2 + 768]
        
        # Reshape for classification
        x = x.view(-1, x.shape[2])  # [batch*seq_len, fc_size]
        
        # Final classification
        x = self.fc(self.dropout(x))  # [batch*seq_len, num_classes]
        
        return x
    

