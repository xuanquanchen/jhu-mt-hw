# -*- coding: utf-8 -*-
"""
Custom loss functions for punctuation restoration
"""

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - alpha_t: weighting factor for class t
    - gamma: focusing parameter (gamma > 0 reduces the loss for well-classified examples)
    - p_t: predicted probability for the true class
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for each class. Can be:
                - None: no weighting
                - float: single weight for all classes
                - list/tensor: weight for each class
            gamma: Focusing parameter. Higher gamma focuses more on hard examples.
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha)
            elif isinstance(alpha, float):
                # Will be set per batch
                pass
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] - logits
            targets: [batch_size] - class indices
        Returns:
            loss: scalar or tensor
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Get predicted probabilities for true class
        p_t = torch.exp(-ce_loss)  # p_t = exp(-CE) = predicted prob for true class
        
        # Compute focal loss
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class weights (combines class weighting and focal loss).
    """
    def __init__(self, class_weights=None, gamma=2.0, reduction='mean'):
        """
        Args:
            class_weights: tensor of shape [num_classes] - weights for each class
            gamma: focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] - logits
            targets: [batch_size] - class indices
        Returns:
            loss: scalar or tensor
        """
        # Compute cross entropy with class weights
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            weight=self.class_weights
        )
        
        # Get predicted probabilities for true class
        p_t = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

