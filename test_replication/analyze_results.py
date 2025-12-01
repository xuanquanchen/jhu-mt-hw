#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析对比实验结果
比较基线模型和 Adapter 模型的 F1、Precision、Recall 等指标
"""

import os
import json
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

def find_latest_model_outputs():
    """找到最新的模型输出目录"""
    baseline_dirs = sorted(glob('outputs/model_baseline_*'), key=os.path.getmtime, reverse=True)
    adapter_dirs = sorted(glob('outputs/model_adapter_*'), key=os.path.getmtime, reverse=True)
    
    baseline_dir = baseline_dirs[0] if baseline_dirs else None
    adapter_dir = adapter_dirs[0] if adapter_dirs else None
    
    return baseline_dir, adapter_dir

def load_progress_csv(csv_path):
    """加载训练进度 CSV 文件"""
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path, sep=';')
    return df

def extract_metrics(df, label_keys):
    """从 DataFrame 中提取指标"""
    if df is None or len(df) == 0:
        return None
    
    # 获取最后一个 epoch 的最后一个 iteration 的结果（最终结果）
    final_row = df.iloc[-1]
    
    metrics = {
        'val_loss': final_row['val_loss'],
        'accuracy': final_row['accuracy'],
        'f1_scores': {},
        'precision_scores': {},
        'recall_scores': {},
        'macro_f1': 0.0,
        'macro_precision': 0.0,
        'macro_recall': 0.0,
    }
    
    # 提取每个类别的 F1、Precision、Recall
    f1_values = []
    precision_values = []
    recall_values = []
    
    for key in label_keys:
        f1_col = f'f1_{key}'
        precision_col = f'precision_{key}'
        recall_col = f'recall_{key}'
        
        if f1_col in final_row:
            f1_val = final_row[f1_col]
            metrics['f1_scores'][key] = f1_val
            f1_values.append(f1_val)
        
        if precision_col in final_row:
            precision_val = final_row[precision_col]
            metrics['precision_scores'][key] = precision_val
            precision_values.append(precision_val)
        
        if recall_col in final_row:
            recall_val = final_row[recall_col]
            metrics['recall_scores'][key] = recall_val
            recall_values.append(recall_val)
    
    # 计算宏平均
    if f1_values:
        metrics['macro_f1'] = np.mean(f1_values)
    if precision_values:
        metrics['macro_precision'] = np.mean(precision_values)
    if recall_values:
        metrics['macro_recall'] = np.mean(recall_values)
    
    return metrics

def load_hyperparameters(json_path):
    """加载超参数"""
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_comparison(baseline_metrics, adapter_metrics, baseline_hparams, adapter_hparams):
    """打印对比结果"""
    print("=" * 100)
    print("模型对比结果")
    print("=" * 100)
    
    # 模型信息
    print("\n【模型配置】")
    print("-" * 100)
    print(f"{'指标':<30} {'基线模型':<35} {'Adapter 模型':<35}")
    print("-" * 100)
    
    if baseline_hparams:
        baseline_type = baseline_hparams.get('model_type', 'Unknown')
        baseline_params = baseline_hparams.get('trainable_params', 0)
        adapter_type = adapter_hparams.get('model_type', 'Unknown') if adapter_hparams else 'Unknown'
        adapter_params = adapter_hparams.get('trainable_params', 0) if adapter_hparams else 0
        adapter_amp = str(adapter_hparams.get('use_amp', False)) if adapter_hparams else 'False'
        print(f"{'模型类型':<30} {baseline_type:<35} {adapter_type:<35}")
        print(f"{'可训练参数':<30} {baseline_params:,} {'':<25} {adapter_params:,} {'':<25}")
        print(f"{'使用 AMP':<30} {str(baseline_hparams.get('use_amp', False)):<35} {adapter_amp:<35}")
    
    # 整体指标
    print("\n【整体性能指标】")
    print("-" * 100)
    print(f"{'指标':<30} {'基线模型':<35} {'Adapter 模型':<35} {'改进':<20}")
    print("-" * 100)
    
    if baseline_metrics and adapter_metrics:
        # Validation Loss
        baseline_loss = baseline_metrics['val_loss']
        adapter_loss = adapter_metrics['val_loss']
        loss_improvement = ((baseline_loss - adapter_loss) / baseline_loss * 100) if baseline_loss > 0 else 0
        print(f"{'Validation Loss':<30} {baseline_loss:.6f} {'':<25} {adapter_loss:.6f} {'':<25} {loss_improvement:+.2f}%")
        
        # Accuracy
        baseline_acc = baseline_metrics['accuracy']
        adapter_acc = adapter_metrics['accuracy']
        acc_improvement = (adapter_acc - baseline_acc) * 100
        print(f"{'Accuracy':<30} {baseline_acc:.4f} {'':<25} {adapter_acc:.4f} {'':<25} {acc_improvement:+.2f}%")
        
        # Macro F1
        baseline_macro_f1 = baseline_metrics['macro_f1']
        adapter_macro_f1 = adapter_metrics['macro_f1']
        f1_improvement = (adapter_macro_f1 - baseline_macro_f1) * 100
        print(f"{'Macro F1':<30} {baseline_macro_f1:.4f} {'':<25} {adapter_macro_f1:.4f} {'':<25} {f1_improvement:+.2f}%")
        
        # Macro Precision
        baseline_macro_precision = baseline_metrics['macro_precision']
        adapter_macro_precision = adapter_metrics['macro_precision']
        precision_improvement = (adapter_macro_precision - baseline_macro_precision) * 100
        print(f"{'Macro Precision':<30} {baseline_macro_precision:.4f} {'':<25} {adapter_macro_precision:.4f} {'':<25} {precision_improvement:+.2f}%")
        
        # Macro Recall
        baseline_macro_recall = baseline_metrics['macro_recall']
        adapter_macro_recall = adapter_metrics['macro_recall']
        recall_improvement = (adapter_macro_recall - baseline_macro_recall) * 100
        print(f"{'Macro Recall':<30} {baseline_macro_recall:.4f} {'':<25} {adapter_macro_recall:.4f} {'':<25} {recall_improvement:+.2f}%")
    
    # 每个类别的详细指标
    if baseline_metrics and adapter_metrics:
        print("\n【每个类别的详细指标】")
        print("-" * 100)
        
        # 获取所有类别
        all_labels = set(list(baseline_metrics['f1_scores'].keys()) + 
                        list(adapter_metrics['f1_scores'].keys()))
        
        for label in sorted(all_labels):
            print(f"\n类别: {label}")
            print(f"  {'指标':<20} {'基线模型':<15} {'Adapter 模型':<15} {'改进':<15}")
            print(f"  {'-'*65}")
            
            baseline_f1 = baseline_metrics['f1_scores'].get(label, 0.0)
            adapter_f1 = adapter_metrics['f1_scores'].get(label, 0.0)
            f1_improvement = (adapter_f1 - baseline_f1) * 100
            print(f"  {'F1 Score':<20} {baseline_f1:.4f} {'':<8} {adapter_f1:.4f} {'':<8} {f1_improvement:+.2f}%")
            
            baseline_precision = baseline_metrics['precision_scores'].get(label, 0.0)
            adapter_precision = adapter_metrics['precision_scores'].get(label, 0.0)
            precision_improvement = (adapter_precision - baseline_precision) * 100
            print(f"  {'Precision':<20} {baseline_precision:.4f} {'':<8} {adapter_precision:.4f} {'':<8} {precision_improvement:+.2f}%")
            
            baseline_recall = baseline_metrics['recall_scores'].get(label, 0.0)
            adapter_recall = adapter_metrics['recall_scores'].get(label, 0.0)
            recall_improvement = (adapter_recall - baseline_recall) * 100
            print(f"  {'Recall':<20} {baseline_recall:.4f} {'':<8} {adapter_recall:.4f} {'':<8} {recall_improvement:+.2f}%")
    
    print("\n" + "=" * 100)

def main():
    print("=" * 100)
    print("分析对比实验结果")
    print("=" * 100)
    
    # 找到最新的模型输出
    baseline_dir, adapter_dir = find_latest_model_outputs()
    
    if not baseline_dir:
        print("错误: 未找到基线模型输出目录")
        print("请先运行: python train_baseline.py --epochs 10")
        return
    
    if not adapter_dir:
        print("错误: 未找到 Adapter 模型输出目录")
        print("请先运行: python train_adapter.py --epochs 10")
        return
    
    print(f"\n基线模型目录: {baseline_dir}")
    print(f"Adapter 模型目录: {adapter_dir}")
    
    # 加载进度文件
    baseline_progress = os.path.join(baseline_dir, 'progress.csv')
    adapter_progress = os.path.join(adapter_dir, 'progress.csv')
    
    baseline_df = load_progress_csv(baseline_progress)
    adapter_df = load_progress_csv(adapter_progress)
    
    if baseline_df is None:
        print(f"错误: 无法加载基线模型进度文件: {baseline_progress}")
        return
    
    if adapter_df is None:
        print(f"错误: 无法加载 Adapter 模型进度文件: {adapter_progress}")
        return
    
    # 加载超参数
    baseline_hparams = load_hyperparameters(os.path.join(baseline_dir, 'hyperparameters.json'))
    adapter_hparams = load_hyperparameters(os.path.join(adapter_dir, 'hyperparameters.json'))
    
    # 获取标签键
    label_keys = ['O', '，', '。', '？']
    if baseline_hparams and 'punctuation_enc' in baseline_hparams:
        label_keys = list(baseline_hparams['punctuation_enc'].keys())
    
    # 提取指标
    baseline_metrics = extract_metrics(baseline_df, label_keys)
    adapter_metrics = extract_metrics(adapter_df, label_keys)
    
    if baseline_metrics is None:
        print("错误: 无法从基线模型数据中提取指标")
        return
    
    if adapter_metrics is None:
        print("错误: 无法从 Adapter 模型数据中提取指标")
        return
    
    # 打印对比结果
    print_comparison(baseline_metrics, adapter_metrics, baseline_hparams, adapter_hparams)
    
    # 保存对比结果到文件
    output_file = 'comparison_results.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        print_comparison(baseline_metrics, adapter_metrics, baseline_hparams, adapter_hparams)
        sys.stdout = old_stdout
    
    print(f"\n对比结果已保存到: {output_file}")

if __name__ == '__main__':
    main()

