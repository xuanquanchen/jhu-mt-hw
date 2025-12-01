# Whisper 标点符号插件使用指南

## 概述

这个项目包含：
1. **训练脚本** (`train_v3.py`) - 训练标点符号恢复模型
2. **测试脚本** (`test.py`) - 在测试集上评估模型
3. **Whisper 插件** (`whisper_plugin.py`) - 为 Whisper 输出添加标点符号
4. **评估脚本** (`evaluate_whisper_plugin.py`) - 评估 Whisper + 插件的端到端性能

## 工作流程

### 步骤 1: 训练模型（如果还没有训练好的模型）

```bash
cd test_replication
python train_v3.py
```

训练完成后，会在 `outputs/` 目录下创建新文件夹，例如：
- `outputs/model_v3_20251112_003011/`
  - `model` - 模型权重文件
  - `hyperparameters.json` - 超参数配置
  - `progress.csv` - 训练进度

### 步骤 2: 在测试集上评估模型（可选）

```bash
python test.py \
    --model_path outputs/model_v3_20251112_003011/model \
    --hyperparameters outputs/model_v3_20251112_003011/hyperparameters.json \
    --test_data data/test \
    --output results_test.csv
```

### 步骤 3: 使用 Whisper 插件

#### 方式 1: 在 Python 代码中使用

```python
import whisper
from whisper_plugin import WhisperPunctuationPlugin

# 1. 初始化插件（使用训练好的模型）
plugin = WhisperPunctuationPlugin(
    output_dir='../outputs/model_v3_20251112_003011',
    model_type='baseline'  # 或 'adapter'
)

# 2. 加载 Whisper 模型
whisper_model = whisper.load_model("base")

# 3. 转写音频
result = whisper_model.transcribe("audio.wav", language="zh")

# 4. 添加标点符号
original_text = result["text"]
text_with_punctuation = plugin.process(original_text)

print(f"原始: {original_text}")
print(f"添加标点: {text_with_punctuation}")
```

#### 方式 2: 命令行使用

```bash
python whisper_plugin.py \
    --output_dir outputs/model_v3_20251112_003011 \
    --model_type baseline \
    --text "你好世界今天天气很好"
```

### 步骤 4: 评估 Whisper + 插件的端到端性能

#### 准备 THCHS30 数据集

THCHS30 数据集通常包含：
- 音频文件（`.wav`）
- 转录文本文件

你需要准备：
1. **音频文件目录**: `/Users/r3ttalynn/Desktop/MT/data_data_thchs30/`
2. **真实标签文件**: 包含每个音频对应的正确转录文本（带标点）

真实标签文件格式（每行一个句子）：
```
你好世界，今天天气很好。
人工智能是未来的发展方向，我们需要不断学习。
```

或者使用 word-level 格式（每行 `word punctuation`）：
```
你好 O
世界 ，
今天 O
天气 O
很好 。
```

#### 运行评估

```bash
python evaluate_whisper_plugin.py \
    --whisper_model base \
    --plugin_output_dir outputs/model_v3_20251112_003011 \
    --plugin_model_type baseline \
    --audio_dir /Users/r3ttalynn/Desktop/MT/data_data_thchs30/ \
    --ground_truth_file /path/to/ground_truth.txt \
    --output results_whisper_plugin.csv \
    --max_samples 100  # 可选：限制样本数用于快速测试
```

#### 评估指标

评估脚本会计算：
1. **文本准确率**: Whisper 转写的文本（移除标点后）与真实文本的匹配度
2. **标点符号 Precision**: 预测的标点符号中，有多少是正确的
3. **标点符号 Recall**: 真实的标点符号中，有多少被正确预测
4. **标点符号 F1-Score**: Precision 和 Recall 的调和平均

## 数据集使用说明

### THCHS30 数据集

THCHS30 是一个中文语音识别数据集。你需要：

1. **确认数据集结构**:
   ```
   data_data_thchs30/
   ├── train/
   │   ├── *.wav (音频文件)
   │   └── *.txt (转录文本)
   ├── test/
   │   ├── *.wav
   │   └── *.txt
   └── ...
   ```

2. **准备真实标签文件**:
   - 从 THCHS30 的转录文本文件中提取所有句子
   - 保存为一个文本文件，每行一个句子（带标点）
   - 确保顺序与音频文件顺序一致

3. **使用测试集还是训练集？**
   - **推荐使用测试集** (`test/`)，因为：
     - 测试集是专门用于评估的
     - 模型没有在测试集上训练过，结果更可靠
     - 测试集通常更小，评估更快

### 快速测试（使用少量样本）

如果你想快速测试流程，可以使用 `--max_samples` 参数：

```bash
python evaluate_whisper_plugin.py \
    --whisper_model base \
    --plugin_output_dir outputs/model_v3_20251112_003011 \
    --plugin_model_type baseline \
    --audio_dir /Users/r3ttalynn/Desktop/MT/data_data_thchs30/test/ \
    --ground_truth_file /path/to/test_ground_truth.txt \
    --output results_quick_test.csv \
    --max_samples 10  # 只评估前 10 个样本
```

## 常见问题

### Q1: 我应该用训练集还是测试集？

**答**: 使用**测试集** (`test/`)。原因：
- 测试集是专门用于评估的
- 模型没有在测试集上训练，结果更可靠
- 测试集通常更小，评估更快

### Q2: 如何准备真实标签文件？

**答**: 
1. 从 THCHS30 数据集的转录文本文件中提取句子
2. 确保每个句子都有正确的标点符号
3. 保存为文本文件，每行一个句子
4. 确保顺序与音频文件顺序一致

### Q3: 评估需要多长时间？

**答**: 取决于：
- 音频文件数量
- 音频文件长度
- Whisper 模型大小（`tiny` 最快，`large` 最慢）
- 是否使用 GPU

建议先用 `--max_samples 10` 快速测试。

### Q4: 如何选择 Whisper 模型大小？

**答**:
- `tiny`: 最快，准确率较低
- `base`: 平衡速度和准确率（推荐）
- `small`: 更好的准确率
- `medium`: 高准确率，较慢
- `large`: 最高准确率，最慢

### Q5: 如何选择插件模型类型？

**答**:
- `baseline`: 全参数微调，性能稳定
- `adapter`: 参数少，速度快，性能相当（推荐）

## 示例工作流

### 完整示例：从训练到评估

```bash
# 1. 训练模型
cd test_replication
python train_v3.py

# 2. 等待训练完成，找到输出目录
# 例如: outputs/model_v3_20251112_003011/

# 3. 在测试集上评估模型（可选）
python test.py \
    --model_path outputs/model_v3_20251112_003011/model \
    --hyperparameters outputs/model_v3_20251112_003011/hyperparameters.json \
    --test_data data/test \
    --output results_test.csv

# 4. 准备 THCHS30 测试集的真实标签文件
# （手动准备或写脚本提取）

# 5. 评估 Whisper + 插件
python evaluate_whisper_plugin.py \
    --whisper_model base \
    --plugin_output_dir outputs/model_v3_20251112_003011 \
    --plugin_model_type baseline \
    --audio_dir /Users/r3ttalynn/Desktop/MT/data_data_thchs30/test/ \
    --ground_truth_file thchs30_test_ground_truth.txt \
    --output results_whisper_plugin.csv
```

## 结果解读

评估完成后，你会得到：
1. **CSV 文件**: 包含每个样本的详细结果
2. **控制台输出**: 包含整体指标摘要

关键指标：
- **文本准确率**: Whisper 转写的准确性（不考虑标点）
- **标点 F1-Score**: 标点符号恢复的整体性能（最重要的指标）

## 下一步

- 查看 `evaluate_with_plots_v1.ipynb` 生成可视化图表
- 尝试不同的 Whisper 模型大小
- 尝试不同的插件模型类型（baseline vs adapter）
- 调整插件模型的超参数并重新训练


