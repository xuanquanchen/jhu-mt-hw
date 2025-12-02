# -*- coding: utf-8 -*-
"""
Training script for BertChineseEmbSlimCNNlstmBert model
** VERSION 3: Class weights + validation data cap (for faster testing)
3 epochs now
remove ！；、
————————————————————————————————————————————————————————————————————————————
** VERSION 2: Computes class weights from data to handle imbalanced classes
** VERSION 1: 1 epoch only, limited training data 10k (first 99,998 lines till a period punctuation)
Full training dataset 274k take 45 hours + (validation time does not include)
Zihan Lyu, Nov 11, 2025

3 Validation checks, so Epoch 1/1 stops at 32% for the first validation

Features:
- Computes class weights from training data
- Limits both training AND validation data for faster testing
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
import torch
from torch import nn, optim
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
import json
import random
from collections import Counter

from transformers import BertTokenizer
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from model import BertChineseEmbSlimCNNlstmBert
from data_utils import load_file, preprocess_data, create_data_loader

# Configuration
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
print(f"Using device: {device}")

# Hyperparameters
SEED = 20
SEQ_LEN = 200
DROPOUT = 0.3
EPOCHS = 10  # Only 1 epoch for quick testing
BATCH_SIZE = 40
LEARNING_RATE = 5e-5
ITERATIONS = 3  # Number of validation checks per epoch
MAX_TRAIN_LINES = 150000  # Limit training data to first 99,998 lines
MAX_VALID_LINES = 50000  # Limit validation data to first 20,816 lines (for faster validation)

# Punctuation encoding (O + 6 Chinese punctuation marks)
PUNCTUATION_ENC = {
    'O': 0,
    '，': 1,  # COMMA
    '。': 2,  # PERIOD
    '？': 3,  # QUESTION
    # '！': 4,  # EXCLAMATION
    # '；': 5,  # SEMICOLON
    # '、': 6,  # ENUMERATION
}


def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_file_limited(filename, max_lines=None):
    """
    Load data file with optional line limit
    """
    with open(filename, 'r', encoding='utf-8') as f:
        if max_lines:
            data = [f.readline() for _ in range(max_lines) if f.readline()]
            # Reset file pointer and read again properly
            f.seek(0)
            data = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                data.append(line)
        else:
            data = f.readlines()
    return data


def compute_class_weights(data, punctuation_enc):
    """
    Compute class weights from data distribution using inverse frequency weighting.
    
    Args:
        data: List of lines, each line is "word punctuation"
        punctuation_enc: Dictionary mapping punctuation to label id
    
    Returns:
        torch.Tensor: Class weights tensor of shape [num_classes]
    """
    # Count punctuation in data
    punc_counts = Counter()
    total_lines = 0
    
    for line in data:
        parts = line.strip().split()
        if len(parts) == 2:
            punc = parts[1]
            if punc in punctuation_enc:
                punc_counts[punc] += 1
                total_lines += 1
    
    # Print class distribution
    print("\n" + "=" * 60)
    print("Class Distribution in Training Data:")
    print("=" * 60)
    for punc in punctuation_enc.keys():
        count = punc_counts.get(punc, 0)
        percentage = (count / total_lines * 100) if total_lines > 0 else 0.0
        print(f"  {punc:3s} (class {punctuation_enc[punc]}): {count:8,} ({percentage:5.2f}%)")
    print(f"  Total: {total_lines:,} lines")
    print("=" * 60)
    
    # Compute inverse frequency weights
    # Formula: weight[i] = total_samples / (num_classes * count[i])
    # This gives higher weight to rarer classes
    weights = []
    num_classes = len(punctuation_enc)
    
    for punc in punctuation_enc.keys():
        count = punc_counts.get(punc, 1)  # Avoid division by zero
        # Inverse frequency: more common = lower weight
        weight = total_lines / (num_classes * count)
        weights.append(weight)
    
    # Normalize so 'O' (most common) has weight 1.0
    o_weight = weights[0]
    weights = [w / o_weight for w in weights]
    
    # Print computed weights
    print("\nComputed Class Weights (normalized so 'O' = 1.0):")
    print("=" * 60)
    for punc, weight in zip(punctuation_enc.keys(), weights):
        print(f"  {punc:3s} (class {punctuation_enc[punc]}): {weight:8.4f}")
    print("=" * 60)
    
    return torch.tensor(weights, dtype=torch.float32)


def validate(model, criterion, data_loader_valid, punctuation_enc):
    """Validate model"""
    model.eval()
    val_losses = []
    val_accs = []
    val_f1s = []
    
    label_keys = list(punctuation_enc.keys())
    label_vals = list(punctuation_enc.values())
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader_valid, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = model(inputs)  # [batch*seq_len, num_classes]
            labels = labels.view(-1)  # [batch*seq_len]
            
            val_loss = criterion(output, labels)
            val_losses.append(val_loss.cpu().item())
            
            y_pred = output.argmax(dim=1).cpu().numpy()
            y_true = labels.cpu().numpy()
            
            val_accs.append(metrics.accuracy_score(y_true, y_pred))
            val_f1s.append(metrics.f1_score(y_true, y_pred, average=None, labels=label_vals))
    
    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_f1 = np.array(val_f1s).mean(axis=0)
    
    return val_loss, val_acc, val_f1, label_keys


def train(model, optimizer, criterion, data_loader_train, data_loader_valid, 
          save_path, punctuation_enc, epochs, iterations):
    """Training loop"""
    print_every = len(data_loader_train) // iterations + 1
    best_val_loss = float('inf')
    best_model_path = None
    
    progress_path = os.path.join(save_path, 'progress.csv')
    with open(progress_path, 'w') as f:
        label_keys = list(punctuation_enc.keys())
        f1_cols = ';'.join(['f1_' + key for key in label_keys])
        f.write(f'time;epoch;iteration;training_loss;val_loss;accuracy;{f1_cols}\n')
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        counter = 0
        iteration = 1
        
        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)  # [batch*seq_len, num_classes]
            labels_flat = labels.view(-1)  # [batch*seq_len]
            
            loss = criterion(output, labels_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            counter += 1
            
            # Validation
            if counter % print_every == 0:
                train_loss = np.mean(train_losses)
                val_loss, val_acc, val_f1, label_keys = validate(
                    model, criterion, data_loader_valid, punctuation_enc
                )
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(save_path, 'model')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"\n✓ Saved best model (val_loss: {val_loss:.4f})")
                
                # Log progress
                f1_vals = ';'.join([f'{val:.4f}' for val in val_f1])
                with open(progress_path, 'a') as f:
                    f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")};'
                           f'{epoch+1};{iteration};{train_loss:.4f};{val_loss:.4f};'
                           f'{val_acc:.4f};{f1_vals}\n')
                
                print(f"\nEpoch: {epoch+1}/{epochs} | Iteration: {iteration}/{iterations}")
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
                print(f"F1 Scores: {f1_vals}")
                
                train_losses = []
                iteration += 1
                model.train()
        
        # Final validation at end of epoch
        train_loss = np.mean(train_losses) if train_losses else 0.0
        val_loss, val_acc, val_f1, label_keys = validate(
            model, criterion, data_loader_valid, punctuation_enc
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_path, 'model')
            torch.save(model.state_dict(), best_model_path)
            print(f"\n✓ Saved best model (val_loss: {val_loss:.4f})")
        
        f1_vals = ';'.join([f'{val:.4f}' for val in val_f1])
        with open(progress_path, 'a') as f:
            f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")};'
                   f'{epoch+1};{iteration};{train_loss:.4f};{val_loss:.4f};'
                   f'{val_acc:.4f};{f1_vals}\n')
        
        print(f"\n=== Epoch {epoch+1} Complete ===")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"F1 Scores: {f1_vals}\n")
    
    return best_model_path


if __name__ == '__main__':
    # Set random seed
    setup_seed(SEED)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('outputs', f'model_v3_{timestamp}')
    os.makedirs(save_path, exist_ok=True)
    
    print("=" * 60)
    print("Chinese Punctuation Restoration - Training (V3)")
    print("=" * 60)
    print(f"Model: BertChineseEmbSlimCNNlstmBert")
    print(f"Output directory: {save_path}")
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    print(f"Max training lines: {MAX_TRAIN_LINES}")
    print(f"Max validation lines: {MAX_VALID_LINES}")
    print("=" * 60)
    
    # Load data (limited for both train and validation)
    print("\nLoading data...")
    print(f"  Training: limited to first {MAX_TRAIN_LINES:,} lines")
    print(f"  Validation: limited to first {MAX_VALID_LINES:,} lines")
    data_train = load_file_limited('data/train', max_lines=MAX_TRAIN_LINES)
    data_valid = load_file_limited('data/valid', max_lines=MAX_VALID_LINES)
    print(f"Train samples: {len(data_train):,} (limited from full dataset)")
    print(f"Valid samples: {len(data_valid):,} (limited from full dataset)")
    
    # Compute class weights from training data
    print("\nComputing class weights from training data...")
    class_weights = compute_class_weights(data_train, PUNCTUATION_ENC)
    class_weights = class_weights.to(device)
    
    # Save class weights to hyperparameters
    class_weights_dict = {punc: float(weight) for punc, weight in zip(PUNCTUATION_ENC.keys(), class_weights.cpu().tolist())}
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', do_lower_case=True)
    print("✓ Tokenizer loaded")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, y_train = preprocess_data(data_train, tokenizer, PUNCTUATION_ENC, SEQ_LEN)
    X_valid, y_valid = preprocess_data(data_valid, tokenizer, PUNCTUATION_ENC, SEQ_LEN)
    print(f"Train sequences: {X_train.shape[0]:,}")
    print(f"Valid sequences: {X_valid.shape[0]:,}")
    
    # Create data loaders
    data_loader_train = create_data_loader(X_train, y_train, shuffle=True, batch_size=BATCH_SIZE)
    data_loader_valid = create_data_loader(X_valid, y_valid, shuffle=False, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\nInitializing model...")
    output_size = len(PUNCTUATION_ENC)
    model = BertChineseEmbSlimCNNlstmBert(SEQ_LEN, output_size, DROPOUT, None)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and loss with class weights
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\n✓ Using weighted CrossEntropyLoss with computed class weights")
    
    # Save hyperparameters (including class weights and limits)
    hyperparameters = {
        'seq_len': SEQ_LEN,
        'dropout': DROPOUT,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'iterations': ITERATIONS,
        'max_train_lines': MAX_TRAIN_LINES,
        'max_valid_lines': MAX_VALID_LINES,
        'punctuation_enc': PUNCTUATION_ENC,
        'class_weights': class_weights_dict,
    }
    with open(os.path.join(save_path, 'hyperparameters.json'), 'w', encoding='utf-8') as f:
        json.dump(hyperparameters, f, indent=2, ensure_ascii=False)
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_model_path = train(
        model, optimizer, criterion,
        data_loader_train, data_loader_valid,
        save_path, PUNCTUATION_ENC,
        EPOCHS, ITERATIONS
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved to: {best_model_path}")
    print("=" * 60)
