# Adapter 模型改进说明

## 概述

本项目实现了基于 Adapter 的模型改进方案，通过冻结 BERT 参数并添加轻量级 Adapter 层来实现高效微调。同时引入了多种改进方法以提升模型性能。

## 主要改进

### 1. Adapter 微调 (Adapter Fine-tuning)

**原理：**
- 冻结预训练 BERT 的所有参数
- 在每个 BERT 输出后添加轻量级 Adapter 层
- 只训练 Adapter 参数和下游任务相关参数

**优势：**
- **参数效率**：可训练参数减少约 75.5%（从 260M 降到 64M）
- **训练速度**：更快，因为大部分参数不需要梯度计算
- **内存效率**：占用更少显存
- **防止过拟合**：冻结的预训练参数保持稳定

**Adapter 架构：**
```
Input → Down Projection (768 → 384) → GELU → Up Projection (384 → 768) → Residual
```

### 2. 注意力融合 (Attention-based Feature Fusion)

**原理：**
- 使用注意力机制动态融合多个特征表示
- 自动学习不同特征的重要性权重

**优势：**
- 比简单拼接更智能
- 能够根据上下文自适应调整特征权重
- 提升模型表达能力

### 3. Focal Loss

**原理：**
- 专门处理类别不平衡问题
- 通过 `(1-p_t)^gamma` 项聚焦难分类样本
- 结合类别权重进一步平衡

**公式：**
```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

**优势：**
- 更好地处理类别不平衡
- 关注难分类样本
- 提升少数类别的性能

## 文件说明

- `model_adapter.py`: Adapter 版本的模型实现
- `losses.py`: 自定义损失函数（Focal Loss）
- `train_adapter.py`: 训练脚本
- `test_adapter_model.py`: 模型测试脚本

## 使用方法

### 1. 激活环境

```bash
source .venv/bin/activate
```

### 2. 训练 Adapter 模型（默认配置）

```bash
python train_adapter.py
```

默认配置：
- ✅ 使用 Adapter（冻结 BERT）
- ✅ 使用注意力融合
- ✅ 使用 Focal Loss (gamma=2.0)
- Epochs: 15
- Batch size: 40
- Learning rate: 1e-4

### 3. 自定义配置

```bash
# 不使用 Adapter（全参数微调）
python train_adapter.py --no-adapter

# 不使用注意力融合
python train_adapter.py --no-attention-fusion

# 不使用 Focal Loss
python train_adapter.py --no-focal-loss

# 调整 Focal Loss gamma
python train_adapter.py --focal-gamma 1.5

# 调整 Adapter 大小
python train_adapter.py --adapter-size 256

# 组合使用
python train_adapter.py --adapter-size 256 --focal-gamma 2.5 --epochs 20
```

### 4. 测试模型

```bash
python test_adapter_model.py
```

## 实验对比

### 参数对比

| 配置 | 总参数 | 可训练参数 | 可训练比例 |
|------|--------|-----------|-----------|
| 原始模型（无 Adapter）| 260M | 260M | 100% |
| Adapter 模型 | 268M | 64M | 23.77% |

**参数减少：75.5%**

### 预期改进

1. **训练速度**：提升约 2-3 倍（因为大部分参数不需要梯度）
2. **内存占用**：减少约 40-50%
3. **性能**：预期保持或略优于原始模型（通过更好的特征融合和损失函数）

## 其他可能的改进方向

1. **更好的预训练模型**：
   - 使用 RoBERTa 或更大型的模型
   - 使用专门针对中文的模型（如 Chinese-BERT-wwm）

2. **数据增强**：
   - 随机删除/替换标点
   - 同义词替换
   - 回译

3. **架构改进**：
   - 使用 Transformer 层替代 LSTM
   - 添加残差连接
   - 使用 Layer Normalization

4. **训练策略**：
   - 学习率调度（warmup + cosine decay）
   - 梯度累积
   - 混合精度训练

5. **集成方法**：
   - 模型集成
   - 投票机制

## 下一步实验建议

1. **基线对比**：运行原始模型和 Adapter 模型，对比性能
2. **消融实验**：
   - Adapter vs 全参数微调
   - 注意力融合 vs 简单拼接
   - Focal Loss vs 加权交叉熵
3. **超参数调优**：
   - Adapter 大小（128, 256, 384, 512）
   - Focal Loss gamma (1.0, 1.5, 2.0, 2.5)
   - 学习率 (1e-5, 5e-5, 1e-4, 5e-4)

## 注意事项

1. Adapter 模型的学习率可以设置得更高（1e-4），因为参数更少
2. 如果显存不足，可以减小 batch size
3. 建议先用小数据集测试，确认模型正常工作后再全量训练

