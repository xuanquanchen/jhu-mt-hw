# -*- coding: utf-8 -*-
"""
Training script for BertChineseEmbSlimCNNlstmBert model
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
DROPOUT = 0.1
EPOCHS = 15
BATCH_SIZE = 10
LEARNING_RATE = 1e-5
ITERATIONS = 3  # Number of validation checks per epoch

# Punctuation encoding (O + 6 Chinese punctuation marks)
PUNCTUATION_ENC = {
    'O': 0,
    '，': 1,  # COMMA
    '。': 2,  # PERIOD
    '？': 3,  # QUESTION
    '！': 4,  # EXCLAMATION
    '；': 5,  # SEMICOLON
    '、': 6,  # ENUMERATION
}


def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
    save_path = os.path.join('outputs', f'model_{timestamp}')
    os.makedirs(save_path, exist_ok=True)
    
    # Save hyperparameters
    hyperparameters = {
        'seq_len': SEQ_LEN,
        'dropout': DROPOUT,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'iterations': ITERATIONS,
        'punctuation_enc': PUNCTUATION_ENC,
    }
    with open(os.path.join(save_path, 'hyperparameters.json'), 'w', encoding='utf-8') as f:
        json.dump(hyperparameters, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("Chinese Punctuation Restoration - Training")
    print("=" * 60)
    print(f"Model: BertChineseEmbSlimCNNlstmBert")
    print(f"Output directory: {save_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data_train = load_file('data/train')
    data_valid = load_file('data/valid')
    print(f"Train samples: {len(data_train)}")
    print(f"Valid samples: {len(data_valid)}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', do_lower_case=True)
    print("✓ Tokenizer loaded")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, y_train = preprocess_data(data_train, tokenizer, PUNCTUATION_ENC, SEQ_LEN)
    X_valid, y_valid = preprocess_data(data_valid, tokenizer, PUNCTUATION_ENC, SEQ_LEN)
    print(f"Train sequences: {X_train.shape[0]}")
    print(f"Valid sequences: {X_valid.shape[0]}")
    
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
    
    # Setup optimizer and loss
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
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

