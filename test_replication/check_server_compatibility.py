#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查服务器兼容性
检查代码是否可以在服务器上运行
"""

import os
import sys
import platform

def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python 版本: {version.major}.{version.minor}.{version.micro} (符合要求)")
        return True
    else:
        print(f"❌ Python 版本: {version.major}.{version.minor}.{version.micro} (需要 3.8+)")
        return False

def check_dependencies():
    """检查依赖"""
    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'sklearn',
        'tqdm',
        'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} (未安装)")
            missing.append(package)
    
    return len(missing) == 0

def check_cuda():
    """检查 CUDA"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA 可用")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 版本: {torch.version.cuda}")
        else:
            print("⚠️  CUDA 不可用（将使用 CPU，训练会很慢）")
        return cuda_available
    except ImportError:
        print("❌ PyTorch 未安装")
        return False

def check_file_paths():
    """检查文件路径（是否有硬编码路径）"""
    print("\n检查文件路径...")
    
    # 检查关键文件
    required_files = [
        'train_baseline.py',
        'train_adapter.py',
        'model.py',
        'data_utils.py',
        'requirements.txt'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file} (缺失)")
            missing.append(file)
    
    return len(missing) == 0

def check_data_directory():
    """检查数据目录"""
    print("\n检查数据目录...")
    if os.path.exists('data'):
        print("✓ data/ 目录存在")
        if os.path.exists('data/train'):
            lines = sum(1 for _ in open('data/train', 'r', encoding='utf-8') if _)
            print(f"✓ data/train 存在 ({lines:,} 行)")
        else:
            print("⚠️  data/train 不存在")
        
        if os.path.exists('data/valid'):
            print("✓ data/valid 存在")
        else:
            print("⚠️  data/valid 不存在")
    else:
        print("⚠️  data/ 目录不存在（需要准备数据）")
    
    return True

def check_relative_paths():
    """检查是否使用相对路径"""
    print("\n检查路径配置...")
    
    # 检查关键文件中的路径
    files_to_check = ['train_baseline.py', 'train_adapter.py']
    issues = []
    
    for file in files_to_check:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 检查是否有绝对路径
                if '/Users/' in content or '/home/' in content or 'C:\\' in content:
                    issues.append(f"{file} 可能包含硬编码路径")
    
    if issues:
        for issue in issues:
            print(f"⚠️  {issue}")
    else:
        print("✓ 未发现硬编码路径（使用相对路径）")
    
    return len(issues) == 0

def main():
    print("=" * 60)
    print("服务器兼容性检查")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print()
    
    results = []
    
    # 检查 Python 版本
    print("1. Python 版本:")
    results.append(check_python_version())
    print()
    
    # 检查依赖
    print("2. 依赖包:")
    results.append(check_dependencies())
    print()
    
    # 检查 CUDA
    print("3. GPU/CUDA:")
    cuda_available = check_cuda()
    print()
    
    # 检查文件
    print("4. 关键文件:")
    results.append(check_file_paths())
    print()
    
    # 检查数据
    check_data_directory()
    print()
    
    # 检查路径
    results.append(check_relative_paths())
    print()
    
    # 总结
    print("=" * 60)
    print("检查总结")
    print("=" * 60)
    
    all_ok = all(results)
    
    if all_ok:
        print("✅ 代码可以在服务器上运行！")
        print()
        print("建议:")
        if cuda_available:
            print("  ✓ 检测到 GPU，训练速度会很快")
        else:
            print("  ⚠️  未检测到 GPU，建议使用 GPU 服务器")
        print("  ✓ 使用 screen/tmux 确保训练不中断")
        print("  ✓ 使用 nohup 或 screen 后台运行")
    else:
        print("⚠️  发现一些问题，请先解决:")
        if not results[0]:
            print("  - 需要 Python 3.8+")
        if not results[1]:
            print("  - 需要安装依赖: pip install -r requirements.txt")
        if not results[2]:
            print("  - 缺少关键文件")
    
    print("=" * 60)
    
    return all_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

