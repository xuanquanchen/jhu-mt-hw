# -*- coding: utf-8 -*-
"""
Training script for Adapter-based model with improvements
Supports:
- Adapter fine-tuning (freeze BERT, train adapters only)
- Focal Loss for class imbalance
- Attention-based feature fusion
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
import torch
from torch import nn, optim
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
from tqdm import tqdm
from datetime import datetime
import json
import random
from collections import Counter
import argparse

from transformers import BertTokenizer
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from model_adapter import BertChineseEmbSlimCNNlstmBertAdapter
from losses import FocalLoss, WeightedFocalLoss
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
BATCH_SIZE = 40
LEARNING_RATE = 1e-4  # Higher LR for adapters (they're smaller)
ITERATIONS = 3
MAX_TRAIN_LINES = 150000
MAX_VALID_LINES = 50000

# Model options
USE_ADAPTER = True
ADAPTER_SIZE = 384  # bert_size // 2
USE_ATTENTION_FUSION = True
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

# Training optimizations
USE_AMP = True  # Mixed precision training (faster, less memory)

# Punctuation encoding
PUNCTUATION_ENC = {
    'O': 0,
    '，': 1,  # COMMA
    '。': 2,  # PERIOD
    '？': 3,  # QUESTION
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
    """Load data file with optional line limit"""
    with open(filename, 'r', encoding='utf-8') as f:
        if max_lines:
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
    """Compute class weights from data distribution"""
    punc_counts = Counter()
    total_lines = 0
    
    for line in data:
        parts = line.strip().split()
        if len(parts) == 2:
            punc = parts[1]
            if punc in punctuation_enc:
                punc_counts[punc] += 1
                total_lines += 1
    
    print("\n" + "=" * 60)
    print("Class Distribution in Training Data:")
    print("=" * 60)
    for punc in punctuation_enc.keys():
        count = punc_counts.get(punc, 0)
        percentage = (count / total_lines * 100) if total_lines > 0 else 0.0
        print(f"  {punc:3s} (class {punctuation_enc[punc]}): {count:8,} ({percentage:5.2f}%)")
    print(f"  Total: {total_lines:,} lines")
    print("=" * 60)
    
    weights = []
    num_classes = len(punctuation_enc)
    
    for punc in punctuation_enc.keys():
        count = punc_counts.get(punc, 1)
        weight = total_lines / (num_classes * count)
        weights.append(weight)
    
    # Normalize so 'O' (most common) has weight 1.0
    o_weight = weights[0]
    weights = [w / o_weight for w in weights]
    
    print("\nComputed Class Weights (normalized so 'O' = 1.0):")
    print("=" * 60)
    for punc, weight in zip(punctuation_enc.keys(), weights):
        print(f"  {punc:3s} (class {punctuation_enc[punc]}): {weight:8.4f}")
    print("=" * 60)
    
    return torch.tensor(weights, dtype=torch.float32)


def validate(model, criterion, data_loader_valid, punctuation_enc, use_amp=False):
    """Validate model"""
    model.eval()
    val_losses = []
    val_accs = []
    val_f1s = []
    val_precisions = []
    val_recalls = []
    
    label_keys = list(punctuation_enc.keys())
    label_vals = list(punctuation_enc.values())
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader_valid, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if use_amp and CUDA:
                with autocast():
                    output = model(inputs)
            else:
                output = model(inputs)
            
            labels = labels.view(-1)
            
            val_loss = criterion(output, labels)
            val_losses.append(val_loss.cpu().item())
            
            y_pred = output.argmax(dim=1).cpu().numpy()
            y_true = labels.cpu().numpy()
            
            val_accs.append(metrics.accuracy_score(y_true, y_pred))
            val_f1s.append(metrics.f1_score(y_true, y_pred, average=None, labels=label_vals, zero_division=0))
            val_precisions.append(metrics.precision_score(y_true, y_pred, average=None, labels=label_vals, zero_division=0))
            val_recalls.append(metrics.recall_score(y_true, y_pred, average=None, labels=label_vals, zero_division=0))
    
    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_f1 = np.array(val_f1s).mean(axis=0)
    val_precision = np.array(val_precisions).mean(axis=0)
    val_recall = np.array(val_recalls).mean(axis=0)
    
    return val_loss, val_acc, val_f1, val_precision, val_recall, label_keys


def train(model, optimizer, criterion, data_loader_train, data_loader_valid, 
          save_path, punctuation_enc, epochs, iterations, use_amp=False):
    """Training loop with optional mixed precision training"""
    print_every = len(data_loader_train) // iterations + 1
    best_val_loss = float('inf')
    best_model_path = None
    
    # Mixed precision training setup
    scaler = GradScaler() if use_amp and CUDA else None
    
    progress_path = os.path.join(save_path, 'progress.csv')
    with open(progress_path, 'w') as f:
        label_keys = list(punctuation_enc.keys())
        f1_cols = ';'.join(['f1_' + key for key in label_keys])
        precision_cols = ';'.join(['precision_' + key for key in label_keys])
        recall_cols = ';'.join(['recall_' + key for key in label_keys])
        f.write(f'time;epoch;iteration;training_loss;val_loss;accuracy;{f1_cols};{precision_cols};{recall_cols}\n')
    
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
            
            # Mixed precision training
            if use_amp and CUDA and scaler is not None:
                with autocast():
                    output = model(inputs)
                    labels_flat = labels.view(-1)
                    loss = criterion(output, labels_flat)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(inputs)
                labels_flat = labels.view(-1)
                loss = criterion(output, labels_flat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            
            train_losses.append(loss.item())
            counter += 1
            
            # Validation
            if counter % print_every == 0:
                train_loss = np.mean(train_losses)
                val_loss, val_acc, val_f1, val_precision, val_recall, label_keys = validate(
                    model, criterion, data_loader_valid, punctuation_enc, use_amp
                )
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(save_path, 'model')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"\n✓ Saved best model (val_loss: {val_loss:.4f})")
                
                # Log progress
                f1_vals = ';'.join([f'{val:.4f}' for val in val_f1])
                precision_vals = ';'.join([f'{val:.4f}' for val in val_precision])
                recall_vals = ';'.join([f'{val:.4f}' for val in val_recall])
                with open(progress_path, 'a') as f:
                    f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")};'
                           f'{epoch+1};{iteration};{train_loss:.4f};{val_loss:.4f};'
                           f'{val_acc:.4f};{f1_vals};{precision_vals};{recall_vals}\n')
                
                print(f"\nEpoch: {epoch+1}/{epochs} | Iteration: {iteration}/{iterations}")
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
                print(f"F1 Scores: {f1_vals}")
                print(f"Precision: {precision_vals}")
                print(f"Recall: {recall_vals}")
                
                train_losses = []
                iteration += 1
                model.train()
        
        # Final validation at end of epoch
        train_loss = np.mean(train_losses) if train_losses else 0.0
        val_loss, val_acc, val_f1, val_precision, val_recall, label_keys = validate(
            model, criterion, data_loader_valid, punctuation_enc, use_amp
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_path, 'model')
            torch.save(model.state_dict(), best_model_path)
            print(f"\n✓ Saved best model (val_loss: {val_loss:.4f})")
        
        f1_vals = ';'.join([f'{val:.4f}' for val in val_f1])
        precision_vals = ';'.join([f'{val:.4f}' for val in val_precision])
        recall_vals = ';'.join([f'{val:.4f}' for val in val_recall])
        with open(progress_path, 'a') as f:
            f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")};'
                   f'{epoch+1};{iteration};{train_loss:.4f};{val_loss:.4f};'
                   f'{val_acc:.4f};{f1_vals};{precision_vals};{recall_vals}\n')
        
        print(f"\n=== Epoch {epoch+1} Complete ===")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"F1 Scores: {f1_vals}")
        print(f"Precision: {precision_vals}")
        print(f"Recall: {recall_vals}\n")
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train Adapter-based model')
    parser.add_argument('--use-adapter', action='store_true', default=True,
                        help='Use adapter layers (default: True)')
    parser.add_argument('--no-adapter', dest='use_adapter', action='store_false',
                        help='Disable adapter layers')
    parser.add_argument('--adapter-size', type=int, default=384,
                        help='Adapter size (default: 384)')
    parser.add_argument('--use-attention-fusion', action='store_true', default=True,
                        help='Use attention-based feature fusion (default: True)')
    parser.add_argument('--no-attention-fusion', dest='use_attention_fusion', action='store_false',
                        help='Disable attention fusion')
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                        help='Use Focal Loss (default: True)')
    parser.add_argument('--no-focal-loss', dest='use_focal_loss', action='store_false',
                        help='Disable Focal Loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal Loss gamma parameter (default: 2.0)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=40,
                        help='Batch size (default: 40)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use mixed precision training (default: True)')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                        help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    # Set random seed
    setup_seed(SEED)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "adapter"
    if args.use_adapter:
        model_name += "_adapter"
    if args.use_attention_fusion:
        model_name += "_attn"
    if args.use_focal_loss:
        model_name += "_focal"
    
    save_path = os.path.join('outputs', f'model_{model_name}_{timestamp}')
    os.makedirs(save_path, exist_ok=True)
    
    print("=" * 60)
    print("Chinese Punctuation Restoration - Adapter Model Training")
    print("=" * 60)
    print(f"Model: BertChineseEmbSlimCNNlstmBertAdapter")
    print(f"Output directory: {save_path}")
    print(f"Device: {device}")
    print(f"Use Adapter: {args.use_adapter}")
    print(f"Adapter Size: {args.adapter_size}")
    print(f"Use Attention Fusion: {args.use_attention_fusion}")
    print(f"Use Focal Loss: {args.use_focal_loss}")
    if args.use_focal_loss:
        print(f"Focal Gamma: {args.focal_gamma}")
    print(f"Use Mixed Precision (AMP): {args.use_amp and CUDA}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Max training lines: {MAX_TRAIN_LINES}")
    print(f"Max validation lines: {MAX_VALID_LINES}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data_train = load_file_limited('data/train', max_lines=MAX_TRAIN_LINES)
    data_valid = load_file_limited('data/valid', max_lines=MAX_VALID_LINES)
    print(f"Train samples: {len(data_train):,}")
    print(f"Valid samples: {len(data_valid):,}")
    
    # Compute class weights
    print("\nComputing class weights from training data...")
    class_weights = compute_class_weights(data_train, PUNCTUATION_ENC)
    class_weights = class_weights.to(device)
    
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
    data_loader_train = create_data_loader(X_train, y_train, shuffle=True, batch_size=args.batch_size)
    data_loader_valid = create_data_loader(X_valid, y_valid, shuffle=False, batch_size=args.batch_size)
    
    # Initialize model
    print("\nInitializing model...")
    output_size = len(PUNCTUATION_ENC)
    model = BertChineseEmbSlimCNNlstmBertAdapter(
        SEQ_LEN, output_size, DROPOUT, None,
        use_adapter=args.use_adapter,
        adapter_size=args.adapter_size,
        use_attention_fusion=args.use_attention_fusion
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Setup optimizer (only trainable parameters)
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=args.lr)
    
    # Setup loss function
    if args.use_focal_loss:
        criterion = WeightedFocalLoss(
            class_weights=class_weights,
            gamma=args.focal_gamma
        )
        print(f"\n✓ Using WeightedFocalLoss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\n✓ Using weighted CrossEntropyLoss")
    
    # Save hyperparameters
    hyperparameters = {
        'model_type': 'BertChineseEmbSlimCNNlstmBertAdapter',
        'seq_len': SEQ_LEN,
        'dropout': DROPOUT,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'iterations': ITERATIONS,
        'max_train_lines': MAX_TRAIN_LINES,
        'max_valid_lines': MAX_VALID_LINES,
        'punctuation_enc': PUNCTUATION_ENC,
        'class_weights': {punc: float(weight) for punc, weight in zip(PUNCTUATION_ENC.keys(), class_weights.cpu().tolist())},
        'use_adapter': args.use_adapter,
        'adapter_size': args.adapter_size,
        'use_attention_fusion': args.use_attention_fusion,
        'use_focal_loss': args.use_focal_loss,
        'focal_gamma': args.focal_gamma if args.use_focal_loss else None,
        'use_amp': args.use_amp and CUDA,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
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
        args.epochs, ITERATIONS,
        use_amp=args.use_amp and CUDA
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved to: {best_model_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

