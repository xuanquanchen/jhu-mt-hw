#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行对比实验：基线模型 vs Adapter 模型
各运行 10 个 epoch，然后对比结果
"""

import os
import subprocess
import sys
from datetime import datetime

def run_command(cmd, description):
    """运行命令并显示输出"""
    print("\n" + "=" * 80)
    print(f"开始: {description}")
    print("=" * 80)
    print(f"命令: {' '.join(cmd)}")
    print("-" * 80)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✓ {description} 完成")
    else:
        print(f"\n✗ {description} 失败 (返回码: {result.returncode})")
        sys.exit(1)
    
    return result

def main():
    # 激活虚拟环境
    venv_python = os.path.join('.venv', 'bin', 'python')
    if not os.path.exists(venv_python):
        print("错误: 虚拟环境不存在，请先运行: uv venv")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"comparison_experiment_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print("=" * 80)
    print("对比实验：基线模型 vs Adapter 模型")
    print("=" * 80)
    print(f"实验目录: {experiment_dir}")
    print(f"每个模型将训练 10 个 epoch")
    print("=" * 80)
    
    # 实验 1: 基线模型
    print("\n\n" + "=" * 80)
    print("实验 1/2: 训练基线模型 (BertChineseEmbSlimCNNlstmBert)")
    print("=" * 80)
    
    baseline_cmd = [
        venv_python,
        'train_baseline.py',
        '--epochs', '10',
        '--batch-size', '40',
        '--lr', '1e-5',
        '--use-amp'
    ]
    
    run_command(baseline_cmd, "基线模型训练")
    
    # 实验 2: Adapter 模型
    print("\n\n" + "=" * 80)
    print("实验 2/2: 训练 Adapter 模型")
    print("=" * 80)
    
    adapter_cmd = [
        venv_python,
        'train_adapter.py',
        '--epochs', '10',
        '--batch-size', '40',
        '--lr', '1e-4',
        '--use-adapter',
        '--adapter-size', '384',
        '--use-attention-fusion',
        '--use-focal-loss',
        '--focal-gamma', '2.0',
        '--use-amp'
    ]
    
    run_command(adapter_cmd, "Adapter 模型训练")
    
    print("\n" + "=" * 80)
    print("所有训练完成！")
    print("=" * 80)
    print("\n请运行以下命令分析结果:")
    print(f"  python analyze_results.py")
    print("\n或者手动查看 outputs/ 目录下的结果文件")
    print("=" * 80)

if __name__ == '__main__':
    main()

