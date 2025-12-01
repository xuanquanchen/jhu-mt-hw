# 优化技巧说明

## 已实现的优化

### 1. ✅ 混合精度训练 (Mixed Precision Training / AMP)

**实现位置：** `train_adapter.py`, `train_baseline.py`

**说明：**
- 使用 PyTorch 的 `torch.cuda.amp` 模块
- 在前向传播中使用 `autocast()` 自动选择 FP16/FP32
- 使用 `GradScaler` 防止梯度下溢

**优势：**
- **训练速度提升**：约 1.5-2 倍（在支持的 GPU 上）
- **显存占用减少**：约 30-50%
- **精度损失极小**：通常可以忽略

**使用方法：**
```bash
# 默认启用（如果 CUDA 可用）
python train_adapter.py --use-amp

# 禁用
python train_adapter.py --no-amp
```

**代码实现：**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler() if use_amp and CUDA else None

# 训练时
with autocast():
    output = model(inputs)
    loss = criterion(output, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### 2. ✅ Adapter 微调

**实现位置：** `model_adapter.py`

**说明：**
- 冻结 BERT 参数，只训练 Adapter 层
- 可训练参数减少 75.5%

**优势：**
- 参数效率高
- 训练速度快
- 显存占用少
- 防止过拟合

---

### 3. ✅ 注意力融合 (Attention-based Feature Fusion)

**实现位置：** `model_adapter.py`

**说明：**
- 使用注意力机制动态融合多个特征表示
- 比简单拼接更智能

---

### 4. ✅ Focal Loss

**实现位置：** `losses.py`, `train_adapter.py`

**说明：**
- 专门处理类别不平衡问题
- 聚焦难分类样本

---

## 评估但未采用的优化

### 1. ❌ 旋转位置编码 (RoPE - Rotary Position Embedding)

**评估结果：不适用**

**原因：**
1. **BERT 已有位置编码**：BERT 使用可学习的位置嵌入，已经能够很好地处理位置信息
2. **架构不匹配**：RoPE 主要用于自回归模型（如 GPT），而 BERT 是双向编码器
3. **实现复杂度高**：需要修改 Transformer 层的注意力计算，改动较大
4. **收益不确定**：在这个任务中，BERT 的位置编码已经足够，RoPE 可能不会带来明显提升

**结论：** 在当前架构中，RoPE 不适用。如果未来改用纯 Transformer 架构（移除 BERT），可以考虑 RoPE。

---

### 2. ❌ Weight Tying (权重共享)

**评估结果：不适用**

**原因：**
1. **任务类型不匹配**：Weight tying 通常用于语言模型的输入和输出嵌入共享
2. **架构差异**：我们的任务是分类任务，输入是 BERT token，输出是标点符号类别，两者语义空间完全不同
3. **无共享意义**：输入 token 嵌入和输出分类层权重没有共享的意义

**结论：** 在当前任务中，weight tying 不适用。它主要用于：
- 语言模型的输入/输出嵌入共享
- 序列到序列模型的编码器/解码器嵌入共享

---

## 其他可能的优化（未来考虑）

### 1. 学习率调度
- Warmup + Cosine Decay
- 可以进一步提升训练稳定性

### 2. 梯度累积
- 在显存受限时，可以模拟更大的 batch size

### 3. 数据增强
- 随机删除/替换标点
- 同义词替换
- 回译

### 4. 更好的预训练模型
- 使用 RoBERTa 或更大型的模型
- 使用专门针对中文的模型

---

## 总结

**已实现的优化：**
- ✅ 混合精度训练 (AMP) - **已实现并默认启用**
- ✅ Adapter 微调 - **已实现**
- ✅ 注意力融合 - **已实现**
- ✅ Focal Loss - **已实现**

**评估结果：**
- ❌ RoPE - **不适用**（BERT 已有位置编码，架构不匹配）
- ❌ Weight Tying - **不适用**（分类任务，输入输出语义空间不同）

**建议：**
当前实现的优化已经涵盖了适合这个任务的技巧。混合精度训练可以显著提升训练速度和减少显存占用，建议保持启用。

