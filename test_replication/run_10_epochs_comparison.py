#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行10轮对比实验：基线模型 vs Adapter 模型
各运行 10 个 epoch，结果保存到日志文件
"""

import os
import subprocess
import sys
from datetime import datetime

def run_training_with_log(script_name, log_file, description, *args):
    """运行训练脚本并保存日志"""
    venv_python = os.path.join('.venv', 'bin', 'python')
    if not os.path.exists(venv_python):
        print("错误: 虚拟环境不存在，请先运行: uv venv")
        sys.exit(1)
    
    cmd = [venv_python, script_name] + list(args)
    
    print(f"\n{'='*80}")
    print(f"开始: {description}")
    print(f"日志文件: {log_file}")
    print(f"{'='*80}")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        # 写入开始信息
        f.write(f"{'='*80}\n")
        f.write(f"{description}\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"命令: {' '.join(cmd)}\n")
        f.write(f"{'='*80}\n\n")
        f.flush()
        
        # 运行命令，同时输出到控制台和文件
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出到控制台和文件
        for line in process.stdout:
            print(line, end='')  # 输出到控制台
            f.write(line)  # 写入文件
            f.flush()
        
        process.wait()
        
        # 写入结束信息
        f.write(f"\n{'='*80}\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"返回码: {process.returncode}\n")
        f.write(f"{'='*80}\n")
    
    if process.returncode == 0:
        print(f"\n✓ {description} 完成")
        print(f"日志已保存到: {log_file}")
    else:
        print(f"\n✗ {description} 失败 (返回码: {process.returncode})")
        print(f"请查看日志文件: {log_file}")
        sys.exit(1)
    
    return process.returncode

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    baseline_log = f"baseline_training_{timestamp}.log"
    adapter_log = f"adapter_training_{timestamp}.log"
    
    print("=" * 80)
    print("对比实验：基线模型 vs Adapter 模型")
    print("=" * 80)
    print(f"基线模型日志: {baseline_log}")
    print(f"Adapter 模型日志: {adapter_log}")
    print(f"每个模型将训练 10 个 epoch")
    print("=" * 80)
    
    # 实验 1: 基线模型
    print("\n\n" + "=" * 80)
    print("实验 1/2: 训练基线模型 (BertChineseEmbSlimCNNlstmBert)")
    print("=" * 80)
    
    run_training_with_log(
        'train_baseline.py',
        baseline_log,
        '基线模型训练 (10 epochs)',
        '--epochs', '10',
        '--batch-size', '40',
        '--lr', '1e-5',
        '--use-amp'
    )
    
    # 实验 2: Adapter 模型
    print("\n\n" + "=" * 80)
    print("实验 2/2: 训练 Adapter 模型")
    print("=" * 80)
    
    run_training_with_log(
        'train_adapter.py',
        adapter_log,
        'Adapter 模型训练 (10 epochs)',
        '--epochs', '10',
        '--batch-size', '40',
        '--lr', '1e-4',
        '--use-adapter',
        '--adapter-size', '384',
        '--use-attention-fusion',
        '--use-focal-loss',
        '--focal-gamma', '2.0',
        '--use-amp'
    )
    
    print("\n" + "=" * 80)
    print("所有训练完成！")
    print("=" * 80)
    print(f"\n日志文件:")
    print(f"  基线模型: {baseline_log}")
    print(f"  Adapter 模型: {adapter_log}")
    print("\n请运行以下命令分析结果:")
    print(f"  python analyze_results.py")
    print("\n或者手动查看 outputs/ 目录下的结果文件")
    print("=" * 80)

if __name__ == '__main__':
    main()

