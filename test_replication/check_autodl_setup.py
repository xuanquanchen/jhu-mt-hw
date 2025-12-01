#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoDL 环境检查脚本
检查 PyTorch、CUDA、依赖包是否正确安装
"""

import torch
import sys

def check_pytorch():
    """检查 PyTorch"""
    print("=" * 60)
    print("PyTorch 检查")
    print("=" * 60)
    
    version = torch.__version__
    print(f"PyTorch 版本: {version}")
    
    # 检查版本是否符合要求
    major, minor = map(int, version.split('.')[:2])
    if major > 1 or (major == 1 and minor >= 12):
        print("✓ PyTorch 版本符合要求 (>= 1.12.0)")
        return True
    else:
        print("❌ PyTorch 版本过低，需要 >= 1.12.0")
        return False

def check_cuda():
    """检查 CUDA"""
    print("\n" + "=" * 60)
    print("CUDA 检查")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  显存: {props.total_memory / 1024**3:.2f} GB")
        
        print("✓ CUDA 配置正常")
        return True
    else:
        print("⚠️  未检测到 GPU/CUDA")
        print("   训练将使用 CPU（会很慢）")
        return False

def check_transformers():
    """检查 transformers"""
    print("\n" + "=" * 60)
    print("Transformers 检查")
    print("=" * 60)
    
    try:
        import transformers
        version = transformers.__version__
        print(f"Transformers 版本: {version}")
        
        # 检查版本
        major, minor = map(int, version.split('.')[:2])
        if major > 4 or (major == 4 and minor >= 24):
            print("✓ Transformers 版本符合要求 (>= 4.24.0)")
            return True
        else:
            print("❌ Transformers 版本过低，需要 >= 4.24.0")
            print("   运行: pip install transformers>=4.24.0")
            return False
    except ImportError:
        print("❌ Transformers 未安装")
        print("   运行: pip install transformers")
        return False

def check_other_dependencies():
    """检查其他依赖"""
    print("\n" + "=" * 60)
    print("其他依赖检查")
    print("=" * 60)
    
    dependencies = {
        'numpy': '1.21.0',
        'sklearn': None,  # scikit-learn
        'tqdm': '4.54.0',
        'pandas': '1.3.0',
    }
    
    all_ok = True
    for package, min_version in dependencies.items():
        try:
            if package == 'sklearn':
                import sklearn
                mod = sklearn
                name = 'scikit-learn'
            else:
                mod = __import__(package)
                name = package
            
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"❌ {name} 未安装")
            all_ok = False
    
    return all_ok

def check_mixed_precision():
    """检查混合精度训练支持"""
    print("\n" + "=" * 60)
    print("混合精度训练检查")
    print("=" * 60)
    
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("✓ 混合精度训练 (AMP) 支持可用")
        
        if torch.cuda.is_available():
            print("✓ 可以在 GPU 上使用混合精度训练")
        else:
            print("⚠️  需要 GPU 才能使用混合精度训练")
        
        return True
    except ImportError:
        print("❌ 混合精度训练不支持（PyTorch 版本可能过低）")
        return False

def main():
    print("=" * 60)
    print("AutoDL 环境检查")
    print("=" * 60)
    print()
    
    results = []
    
    # 检查各项
    results.append(check_pytorch())
    results.append(check_cuda())
    results.append(check_transformers())
    results.append(check_other_dependencies())
    results.append(check_mixed_precision())
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    if all(results):
        print("✅ 所有检查通过！环境配置正确。")
        print("\n可以开始训练:")
        print("  python run_10_epochs_comparison.py")
    else:
        print("⚠️  发现一些问题，请先解决:")
        if not results[0]:
            print("  - PyTorch 版本需要 >= 1.12.0")
        if not results[1]:
            print("  - 未检测到 GPU（训练会很慢）")
        if not results[2]:
            print("  - Transformers 需要安装或升级")
        if not results[3]:
            print("  - 部分依赖包缺失，运行: pip install -r requirements.txt")
        if not results[4]:
            print("  - 混合精度训练不支持")
    
    print("=" * 60)
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

