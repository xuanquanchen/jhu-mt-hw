# 对比实验说明

## 快速开始

### 1. 运行对比实验（自动运行两个模型各 10 个 epoch）

```bash
# 激活环境
source .venv/bin/activate

# 运行对比实验
python run_comparison_experiment.py
```

这个脚本会：
1. 训练基线模型（10 个 epoch）
2. 训练 Adapter 模型（10 个 epoch）
3. 所有结果保存在 `outputs/` 目录

### 2. 分析结果

```bash
python analyze_results.py
```

这个脚本会：
- 自动找到最新的基线模型和 Adapter 模型结果
- 对比 F1、Precision、Recall 等指标
- 显示每个类别的详细性能
- 保存对比结果到 `comparison_results.txt`

---

## 手动运行（分别训练）

### 训练基线模型

```bash
python train_baseline.py \
    --epochs 10 \
    --batch-size 40 \
    --lr 1e-5 \
    --use-amp
```

### 训练 Adapter 模型

```bash
python train_adapter.py \
    --epochs 10 \
    --batch-size 40 \
    --lr 1e-4 \
    --use-adapter \
    --adapter-size 384 \
    --use-attention-fusion \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --use-amp
```

---

## 输出文件说明

### 训练输出目录结构

```
outputs/
├── model_baseline_YYYYMMDD_HHMMSS/
│   ├── model                    # 最佳模型权重
│   ├── progress.csv             # 训练进度（包含 F1, Precision, Recall）
│   └── hyperparameters.json     # 超参数配置
│
└── model_adapter_YYYYMMDD_HHMMSS/
    ├── model
    ├── progress.csv
    └── hyperparameters.json
```

### progress.csv 格式

```csv
time;epoch;iteration;training_loss;val_loss;accuracy;f1_O;f1_，;f1_。;f1_？;precision_O;precision_，;precision_。;precision_？;recall_O;recall_，;recall_。;recall_？
```

---

## 对比指标说明

### 整体指标
- **Validation Loss**: 验证集损失（越低越好）
- **Accuracy**: 整体准确率
- **Macro F1**: 所有类别的 F1 分数平均值
- **Macro Precision**: 所有类别的精确率平均值
- **Macro Recall**: 所有类别的召回率平均值

### 每个类别的指标
- **F1 Score**: F1 分数（精确率和召回率的调和平均）
- **Precision**: 精确率（预测为正例中真正为正例的比例）
- **Recall**: 召回率（真正例中被正确预测的比例）

---

## 优化技巧总结

### ✅ 已实现并启用
1. **混合精度训练 (AMP)** - 默认启用，提升训练速度 1.5-2 倍
2. **Adapter 微调** - 减少 75.5% 可训练参数
3. **注意力融合** - 智能特征融合
4. **Focal Loss** - 处理类别不平衡

### ❌ 评估后未采用
1. **RoPE (旋转位置编码)** - BERT 已有位置编码，不适用
2. **Weight Tying** - 分类任务不适用

详细说明请参考 `OPTIMIZATION_TECHNIQUES.md`

---

## 预期结果

### 参数对比
- **基线模型**: ~260M 可训练参数
- **Adapter 模型**: ~64M 可训练参数（减少 75.5%）

### 性能预期
- **训练速度**: Adapter 模型预计快 2-3 倍
- **显存占用**: Adapter 模型预计减少 40-50%
- **模型性能**: 预期保持或略优于基线模型

---

## 注意事项

1. **GPU 要求**: 混合精度训练需要支持 FP16 的 GPU（如 V100, RTX 系列）
2. **训练时间**: 每个模型 10 个 epoch 可能需要数小时，取决于硬件
3. **数据限制**: 当前使用 150k 训练样本和 50k 验证样本（可调整）

---

## 故障排除

### 如果找不到模型输出
- 检查 `outputs/` 目录是否存在
- 确认训练已完成（没有中途中断）

### 如果分析脚本报错
- 确认 `progress.csv` 文件存在且格式正确
- 检查是否有足够的训练数据

### 如果显存不足
- 减小 batch size: `--batch-size 20`
- 禁用混合精度: `--no-amp`（不推荐）

