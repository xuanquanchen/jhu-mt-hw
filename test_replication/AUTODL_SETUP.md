# AutoDL 框架选择指南

## 推荐配置

### ✅ 最佳选择（推荐）

**PyTorch 2.0.1 + CUDA 11.8**
- **镜像名称**: `PyTorch 2.0.1` 或 `PyTorch 2.0.1 (CUDA 11.8)`
- **CUDA 版本**: 11.8
- **PyTorch 版本**: 2.0.1
- **兼容性**: ✅ 完全兼容
- **优势**: 
  - 稳定可靠
  - 支持混合精度训练（AMP）
  - 性能优秀
  - 与 transformers 库兼容性好

### ✅ 备选方案

**PyTorch 2.1.0 + CUDA 11.8**
- **镜像名称**: `PyTorch 2.1.0 (CUDA 11.8)`
- **兼容性**: ✅ 完全兼容
- **优势**: 较新版本，可能有性能优化

**PyTorch 1.13.1 + CUDA 11.7**
- **镜像名称**: `PyTorch 1.13.1 (CUDA 11.7)`
- **兼容性**: ✅ 完全兼容（满足 torch>=1.12.0）
- **优势**: 稳定，兼容性好

## 项目依赖要求

根据 `requirements.txt`:
```
torch>=1.12.0,<3.0.0
transformers>=4.24.0,<5.0.0
```

**所有推荐的框架版本都满足这些要求！**

## AutoDL 选择步骤

### 1. 租用实例时选择镜像

在 AutoDL 租用页面：

1. **选择 GPU**: RTX 3090, A100, 或其他支持 CUDA 11.x 的 GPU
2. **选择镜像**: 
   - 搜索 "PyTorch 2.0.1" 或 "PyTorch 2.0.1 CUDA 11.8"
   - 或选择 "PyTorch 2.1.0 CUDA 11.8"
3. **确认 CUDA 版本**: 确保 CUDA 版本 >= 11.1（推荐 11.7 或 11.8）

### 2. 镜像选择界面示例

在 AutoDL 镜像选择界面，你会看到类似选项：

```
推荐镜像:
├── PyTorch 2.0.1 (CUDA 11.8)  ⭐ 推荐
├── PyTorch 2.1.0 (CUDA 11.8)
├── PyTorch 1.13.1 (CUDA 11.7)
└── PyTorch 1.12.0 (CUDA 11.6)
```

**选择**: `PyTorch 2.0.1 (CUDA 11.8)` ⭐

## 验证安装

连接到服务器后，运行以下命令验证：

```bash
# 1. 检查 Python 版本
python --version
# 应该显示: Python 3.8, 3.9, 或 3.10

# 2. 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 预期输出:
# PyTorch: 2.0.1
# CUDA available: True
# CUDA version: 11.8

# 3. 检查 GPU
nvidia-smi
# 应该显示 GPU 信息
```

## 安装项目依赖

AutoDL 镜像通常已预装 PyTorch，但需要安装其他依赖：

```bash
# 1. 进入项目目录
cd test_replication

# 2. 创建虚拟环境（可选，但推荐）
python -m venv .venv
source .venv/bin/activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证 transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 常见问题

### Q1: 如果镜像中没有 PyTorch 2.0.1 怎么办？

**A**: 可以选择：
1. **PyTorch 2.1.0** - 也完全兼容
2. **PyTorch 1.13.1** - 满足最低要求（>=1.12.0）
3. **基础镜像** - 然后手动安装：
   ```bash
   pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Q2: CUDA 版本不匹配怎么办？

**A**: 
- 确保 GPU 支持 CUDA 11.x（大多数现代 GPU 都支持）
- 如果 GPU 只支持 CUDA 10.x，选择对应的 PyTorch 1.12.0 镜像
- 如果 GPU 支持 CUDA 12.x，可以选择 PyTorch 2.1.0+ CUDA 12.x

### Q3: 如何确认 GPU 和 CUDA 兼容性？

**A**: 运行以下命令：
```bash
nvidia-smi
# 查看 CUDA Version（驱动支持的最高版本）

python -c "import torch; print(torch.cuda.is_available())"
# 应该返回 True
```

### Q4: 混合精度训练需要什么版本？

**A**: 
- PyTorch >= 1.6.0 就支持混合精度训练
- 所有推荐的版本都支持
- 代码中使用的 `torch.cuda.amp` 在所有推荐版本中都可用

## 快速检查脚本

创建 `check_autodl_setup.py`:

```python
import torch
import sys

print("=" * 60)
print("AutoDL 环境检查")
print("=" * 60)

# 检查 PyTorch
print(f"PyTorch 版本: {torch.__version__}")
if torch.__version__ < "1.12.0":
    print("❌ PyTorch 版本过低，需要 >= 1.12.0")
    sys.exit(1)
else:
    print("✓ PyTorch 版本符合要求")

# 检查 CUDA
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print("✓ CUDA 配置正常")
else:
    print("⚠️  未检测到 GPU")

# 检查 transformers
try:
    import transformers
    print(f"Transformers 版本: {transformers.__version__}")
    if transformers.__version__ < "4.24.0":
        print("❌ Transformers 版本过低，需要 >= 4.24.0")
        sys.exit(1)
    else:
        print("✓ Transformers 版本符合要求")
except ImportError:
    print("❌ Transformers 未安装")
    sys.exit(1)

print("=" * 60)
print("✅ 环境检查通过！")
print("=" * 60)
```

## 总结

### 推荐配置（最佳）

| 项目 | 推荐值 |
|------|--------|
| **镜像** | PyTorch 2.0.1 (CUDA 11.8) ⭐ |
| **PyTorch** | 2.0.1 |
| **CUDA** | 11.8 |
| **Python** | 3.8-3.10 |
| **GPU** | RTX 3090, A100, 或其他支持 CUDA 11.x 的 GPU |

### 为什么选择 PyTorch 2.0.1？

1. ✅ **稳定可靠**: 经过充分测试，bug 较少
2. ✅ **性能优秀**: 支持最新的优化特性
3. ✅ **兼容性好**: 与 transformers 4.24.0+ 完美兼容
4. ✅ **功能完整**: 支持混合精度训练、所有需要的功能
5. ✅ **广泛使用**: 社区支持好，问题容易解决

### 快速开始

1. 在 AutoDL 选择: **PyTorch 2.0.1 (CUDA 11.8)** 镜像
2. 租用 GPU 实例（推荐 RTX 3090 或 A100）
3. 上传代码和数据
4. 运行 `./deploy_to_server.sh` 或手动安装依赖
5. 开始训练！

