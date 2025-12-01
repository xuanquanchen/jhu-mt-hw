# 训练状态和输出位置

## 当前训练状态

### 基线模型训练
- **状态**: 进行中
- **当前进度**: 第1个epoch，已完成 47% (9/19 批次)
- **开始时间**: 2025-12-01 00:31:39
- **日志文件**: `baseline_training_20251201_003134.log`

### Adapter 模型训练
- **状态**: 等待中（基线模型完成后自动开始）
- **日志文件**: 将在基线模型完成后创建

## 预计完成时间

### 基线模型
- **每个批次**: 约 60-70 秒
- **每个epoch**: 约 20-25 分钟（19个批次 + 3次验证）
- **10个epoch**: 约 **3.5-4 小时**
- **预计完成时间**: 约 04:00-04:30 (从开始时间计算)

### Adapter 模型
- **每个批次**: 约 40-50 秒（参数更少，更快）
- **每个epoch**: 约 15-18 分钟
- **10个epoch**: 约 **2.5-3 小时**
- **预计完成时间**: 约 06:30-07:30 (从开始时间计算)

### 总计
- **总训练时间**: 约 **5.5-7 小时**
- **预计全部完成**: 约 **07:00-07:30**

## 输出位置

### 1. 训练日志文件
```
baseline_training_20251201_003134.log    # 基线模型训练日志
adapter_training_YYYYMMDD_HHMMSS.log     # Adapter模型训练日志（完成后创建）
```

### 2. 模型输出目录
```
outputs/
├── model_baseline_20251201_003139/      # 基线模型输出
│   ├── model                            # 最佳模型权重
│   ├── progress.csv                    # 训练进度（F1, Precision, Recall）
│   └── hyperparameters.json            # 超参数配置
│
└── model_adapter_YYYYMMDD_HHMMSS/      # Adapter模型输出（完成后创建）
    ├── model
    ├── progress.csv
    └── hyperparameters.json
```

### 3. 实时查看训练进度

```bash
# 查看基线模型训练进度
tail -f baseline_training_20251201_003134.log

# 或使用状态检查脚本
python check_training_status.py
```

### 4. 训练完成后

训练完成后，可以：

1. **分析结果**:
   ```bash
   python analyze_results.py
   ```

2. **保存模型用于部署**:
   ```bash
   # 基线模型
   python save_model_for_deployment.py \
       --output_dir outputs/model_baseline_20251201_003139 \
       --deployment_dir models/baseline_model \
       --model_type baseline
   
   # Adapter模型
   python save_model_for_deployment.py \
       --output_dir outputs/model_adapter_YYYYMMDD_HHMMSS \
       --deployment_dir models/adapter_model \
       --model_type adapter
   ```

3. **测试插件**:
   ```bash
   python whisper_plugin.py \
       --output_dir models/adapter_model \
       --model_type adapter \
       --text "测试文本"
   ```

## 文件说明

### progress.csv 格式
包含每个验证点的详细指标：
- `time`: 时间戳
- `epoch`: epoch 编号
- `iteration`: 迭代编号
- `training_loss`: 训练损失
- `val_loss`: 验证损失
- `accuracy`: 准确率
- `f1_O`, `f1_，`, `f1_。`, `f1_？`: 每个类别的 F1 分数
- `precision_O`, `precision_，`, ...: 每个类别的精确率
- `recall_O`, `recall_，`, ...: 每个类别的召回率

### hyperparameters.json
包含所有训练配置：
- 模型类型
- 超参数（学习率、batch size等）
- 类别权重
- 参数统计

## 注意事项

1. **训练不会中断**: 即使关闭终端，训练也会继续（使用 nohup）
2. **检查进度**: 可以随时运行 `python check_training_status.py` 查看进度
3. **日志文件**: 所有输出都保存在日志文件中，可以随时查看
4. **模型保存**: 每个验证点都会保存最佳模型到 `outputs/模型目录/model`

## 当前进度估算

根据当前进度（第1个epoch的47%），预计：
- **基线模型完成**: 约 04:00-04:30
- **Adapter模型完成**: 约 06:30-07:30
- **全部完成**: 约 07:00-07:30

