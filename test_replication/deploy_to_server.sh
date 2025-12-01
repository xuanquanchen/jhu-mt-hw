#!/bin/bash
# 服务器快速部署脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "服务器部署脚本"
echo "=========================================="

# 1. 检查 Python
echo ""
echo "1. 检查 Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3"
    echo "请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Python 版本: $PYTHON_VERSION"

# 2. 检查是否在项目目录
if [ ! -f "requirements.txt" ]; then
    echo "❌ 错误: 未找到 requirements.txt"
    echo "请确保在项目根目录运行此脚本"
    exit 1
fi

# 3. 创建虚拟环境
echo ""
echo "2. 创建虚拟环境..."
if [ -d ".venv" ]; then
    echo "⚠️  虚拟环境已存在，跳过创建"
else
    python3 -m venv .venv
    echo "✓ 虚拟环境已创建"
fi

# 4. 激活虚拟环境
source .venv/bin/activate
echo "✓ 虚拟环境已激活"

# 5. 升级 pip
echo ""
echo "3. 升级 pip..."
pip install --upgrade pip -q
echo "✓ pip 已升级"

# 6. 安装依赖
echo ""
echo "4. 安装依赖..."
echo "这可能需要几分钟..."
pip install -r requirements.txt
echo "✓ 依赖已安装"

# 7. 检查数据
echo ""
echo "5. 检查数据文件..."
if [ ! -d "data" ]; then
    echo "⚠️  警告: data/ 目录不存在"
    echo "   请确保 data/train, data/valid, data/test 存在"
elif [ ! -f "data/train" ]; then
    echo "⚠️  警告: data/train 文件不存在"
    echo "   请准备训练数据"
else
    TRAIN_LINES=$(wc -l < data/train 2>/dev/null || echo "0")
    echo "✓ 训练数据: $TRAIN_LINES 行"
fi

# 8. 检查 GPU
echo ""
echo "6. 检查 GPU..."
python3 -c "
import torch
cuda_available = torch.cuda.is_available()
print('CUDA 可用:', cuda_available)
if cuda_available:
    print('GPU 数量:', torch.cuda.device_count())
    print('GPU 名称:', torch.cuda.get_device_name(0))
else:
    print('⚠️  未检测到 GPU，将使用 CPU 训练（会很慢）')
" || echo "⚠️  无法检查 GPU（可能 PyTorch 未正确安装）"

# 9. 创建必要的目录
echo ""
echo "7. 创建输出目录..."
mkdir -p outputs
echo "✓ 输出目录已创建"

# 10. 检查关键文件
echo ""
echo "8. 检查关键文件..."
REQUIRED_FILES=("train_baseline.py" "train_adapter.py" "model.py" "data_utils.py")
MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "❌ $file (缺失)"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo ""
    echo "❌ 错误: 缺少必要的文件"
    exit 1
fi

# 11. 完成
echo ""
echo "=========================================="
echo "✅ 部署完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 确保数据文件已准备好（data/train, data/valid, data/test）"
echo "2. 运行训练:"
echo "   source .venv/bin/activate"
echo "   python run_10_epochs_comparison.py"
echo ""
echo "或使用 screen/tmux 后台运行:"
echo "   screen -S training"
echo "   python run_10_epochs_comparison.py"
echo "   # 按 Ctrl+A 然后 D 退出"
echo ""
echo "查看训练状态:"
echo "   python check_training_status.py"
echo "=========================================="

