#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查训练状态和日志文件
"""

import os
import glob
from datetime import datetime

def get_latest_log_files():
    """获取最新的日志文件"""
    baseline_logs = sorted(glob.glob('baseline_training_*.log'), key=os.path.getmtime, reverse=True)
    adapter_logs = sorted(glob.glob('adapter_training_*.log'), key=os.path.getmtime, reverse=True)
    
    return baseline_logs[0] if baseline_logs else None, adapter_logs[0] if adapter_logs else None

def get_file_size_mb(filepath):
    """获取文件大小（MB）"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)

def get_file_lines(filepath):
    """获取文件行数"""
    if not os.path.exists(filepath):
        return 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except:
        return 0

def get_last_lines(filepath, n=10):
    """获取最后n行"""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) > n else lines
    except:
        return []

def main():
    print("=" * 80)
    print("训练状态检查")
    print("=" * 80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    baseline_log, adapter_log = get_latest_log_files()
    
    # 基线模型状态
    print("【基线模型训练状态】")
    print("-" * 80)
    if baseline_log:
        size_mb = get_file_size_mb(baseline_log)
        lines = get_file_lines(baseline_log)
        mtime = datetime.fromtimestamp(os.path.getmtime(baseline_log))
        
        print(f"日志文件: {baseline_log}")
        print(f"文件大小: {size_mb:.2f} MB")
        print(f"行数: {lines:,}")
        print(f"最后更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"状态: {'进行中' if (datetime.now() - mtime).seconds < 300 else '可能已完成或暂停'}")
        
        print("\n最后10行输出:")
        print("-" * 80)
        last_lines = get_last_lines(baseline_log, 10)
        for line in last_lines:
            print(line.rstrip())
    else:
        print("日志文件: 未找到（训练可能还未开始）")
    
    print("\n" + "=" * 80)
    
    # Adapter模型状态
    print("【Adapter 模型训练状态】")
    print("-" * 80)
    if adapter_log:
        size_mb = get_file_size_mb(adapter_log)
        lines = get_file_lines(adapter_log)
        mtime = datetime.fromtimestamp(os.path.getmtime(adapter_log))
        
        print(f"日志文件: {adapter_log}")
        print(f"文件大小: {size_mb:.2f} MB")
        print(f"行数: {lines:,}")
        print(f"最后更新: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"状态: {'进行中' if (datetime.now() - mtime).seconds < 300 else '可能已完成或暂停'}")
        
        print("\n最后10行输出:")
        print("-" * 80)
        last_lines = get_last_lines(adapter_log, 10)
        for line in last_lines:
            print(line.rstrip())
    else:
        print("日志文件: 未找到（基线模型训练完成后才会开始）")
    
    print("\n" + "=" * 80)
    print("提示: 使用 'tail -f <日志文件>' 可以实时查看训练进度")
    print("=" * 80)

if __name__ == '__main__':
    main()

