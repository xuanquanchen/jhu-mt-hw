# -*- coding: utf-8 -*-
"""
Test/Evaluation script for BertChineseEmbSlimCNNlstmBert model
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import json
import pandas as pd

from transformers import BertTokenizer
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

from model import BertChineseEmbSlimCNNlstmBert
from data_utils import load_file, preprocess_data, create_data_loader

# Configuration
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
print(f"Using device: {device}")

# Default punctuation encoding (should match training)
PUNCTUATION_ENC = {
    'O': 0,
    '，': 1,  # COMMA
    '。': 2,  # PERIOD
    '？': 3,  # QUESTION
    '！': 4,  # EXCLAMATION
    '；': 5,  # SEMICOLON
    '、': 6,  # ENUMERATION
}

# Reverse mapping for decoding
ID_TO_PUNCTUATION = {v: k for k, v in PUNCTUATION_ENC.items()}


def load_model(model_path, hyperparameters_path, device):
    """Load trained model"""
    # Load hyperparameters
    with open(hyperparameters_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)
    
    seq_len = hyperparams['seq_len']
    dropout = hyperparams['dropout']
    punctuation_enc = hyperparams['punctuation_enc']
    output_size = len(punctuation_enc)
    
    # Initialize model
    model = BertChineseEmbSlimCNNlstmBert(seq_len, output_size, dropout, None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, punctuation_enc, hyperparams


def evaluate(model, data_loader, punctuation_enc, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    label_keys = list(punctuation_enc.keys())
    label_vals = list(punctuation_enc.values())
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = model(inputs)  # [batch*seq_len, num_classes]
            labels_flat = labels.view(-1)  # [batch*seq_len]
            
            preds = output.argmax(dim=1).cpu().numpy()
            labels_np = labels_flat.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    f1_scores = metrics.f1_score(all_labels, all_preds, average=None, labels=label_vals)
    precision_scores = metrics.precision_score(all_labels, all_preds, average=None, labels=label_vals, zero_division=0)
    recall_scores = metrics.recall_score(all_labels, all_preds, average=None, labels=label_vals, zero_division=0)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Punctuation': label_keys,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-Score': f1_scores
    })
    
    return {
        'accuracy': accuracy,
        'f1_scores': f1_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'results_df': results,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'label_keys': label_keys
    }


def print_results(results):
    """Print evaluation results"""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}\n")
    
    print("Per-Class Metrics:")
    print("-" * 60)
    print(results['results_df'].to_string(index=False))
    print("-" * 60)
    
    # Macro and micro averages
    macro_f1 = np.mean(results['f1_scores'])
    macro_precision = np.mean(results['precision_scores'])
    macro_recall = np.mean(results['recall_scores'])
    
    print(f"\nMacro-Averaged Metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    
    # Weighted average (excluding 'O' class)
    non_o_indices = [i for i, key in enumerate(results['label_keys']) if key != 'O']
    if non_o_indices:
        weighted_f1 = np.mean([results['f1_scores'][i] for i in non_o_indices])
        print(f"\nPunctuation-Only F1-Score: {weighted_f1:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Chinese Punctuation Restoration Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--hyperparameters', type=str, required=True,
                       help='Path to hyperparameters.json file')
    parser.add_argument('--test_data', type=str, default='data/test',
                       help='Path to test data file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results (optional)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chinese Punctuation Restoration - Testing")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, punctuation_enc, hyperparams = load_model(
        args.model_path, args.hyperparameters, device
    )
    print("✓ Model loaded")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', do_lower_case=True)
    print("✓ Tokenizer loaded")
    
    # Load and preprocess test data
    print(f"\nLoading test data from {args.test_data}...")
    data_test = load_file(args.test_data)
    print(f"Test samples: {len(data_test)}")
    
    print("\nPreprocessing data...")
    seq_len = hyperparams['seq_len']
    X_test, y_test = preprocess_data(data_test, tokenizer, punctuation_enc, seq_len)
    print(f"Test sequences: {X_test.shape[0]}")
    
    # Create data loader
    data_loader_test = create_data_loader(
        X_test, y_test, shuffle=False, batch_size=args.batch_size
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)
    
    results = evaluate(model, data_loader_test, punctuation_enc, device)
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.output:
        results['results_df'].to_csv(args.output, index=False)
        print(f"\n✓ Results saved to {args.output}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)

