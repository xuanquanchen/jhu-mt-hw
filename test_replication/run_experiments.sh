#!/bin/bash
# 运行对比实验脚本

# 激活环境
source .venv/bin/activate

# 创建实验输出目录
EXPERIMENT_DIR="experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXPERIMENT_DIR

echo "=========================================="
echo "开始运行对比实验"
echo "输出目录: $EXPERIMENT_DIR"
echo "=========================================="

# 实验1: 原始模型（无 Adapter，无注意力融合，无 Focal Loss）
echo ""
echo "实验1: 原始模型（基线）"
echo "----------------------------------------"
python train_adapter.py \
    --no-adapter \
    --no-attention-fusion \
    --no-focal-loss \
    --epochs 3 \
    --batch-size 40 \
    --lr 1e-5 \
    2>&1 | tee $EXPERIMENT_DIR/exp1_baseline.log

# 实验2: Adapter 模型（默认配置）
echo ""
echo "实验2: Adapter 模型（默认配置）"
echo "----------------------------------------"
python train_adapter.py \
    --use-adapter \
    --adapter-size 384 \
    --use-attention-fusion \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --epochs 3 \
    --batch-size 40 \
    --lr 1e-4 \
    2>&1 | tee $EXPERIMENT_DIR/exp2_adapter_default.log

# 实验3: Adapter + 注意力融合（无 Focal Loss）
echo ""
echo "实验3: Adapter + 注意力融合（无 Focal Loss）"
echo "----------------------------------------"
python train_adapter.py \
    --use-adapter \
    --adapter-size 384 \
    --use-attention-fusion \
    --no-focal-loss \
    --epochs 3 \
    --batch-size 40 \
    --lr 1e-4 \
    2>&1 | tee $EXPERIMENT_DIR/exp3_adapter_attn.log

# 实验4: Adapter + Focal Loss（无注意力融合）
echo ""
echo "实验4: Adapter + Focal Loss（无注意力融合）"
echo "----------------------------------------"
python train_adapter.py \
    --use-adapter \
    --adapter-size 384 \
    --no-attention-fusion \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --epochs 3 \
    --batch-size 40 \
    --lr 1e-4 \
    2>&1 | tee $EXPERIMENT_DIR/exp4_adapter_focal.log

# 实验5: 不同 Adapter 大小
echo ""
echo "实验5: Adapter 大小 = 256"
echo "----------------------------------------"
python train_adapter.py \
    --use-adapter \
    --adapter-size 256 \
    --use-attention-fusion \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --epochs 3 \
    --batch-size 40 \
    --lr 1e-4 \
    2>&1 | tee $EXPERIMENT_DIR/exp5_adapter_256.log

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "结果保存在: $EXPERIMENT_DIR"
echo "=========================================="

