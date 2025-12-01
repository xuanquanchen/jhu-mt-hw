# Whisper 插件快速开始指南

## 快速回答你的问题

### 1. 我应该用哪个数据集？

**使用 THCHS30 的测试集 (`test/`)**，不是训练集。

原因：
- 测试集是专门用于评估的
- 模型没有在测试集上训练，结果更可靠
- 测试集通常更小，评估更快

### 2. 工作流程是什么？

```
训练模型 → 使用 Whisper 转写 → 插件添加标点 → 评估准确率
```

### 3. 具体步骤

#### 步骤 1: 确认你有训练好的模型

检查 `outputs/` 目录，找到最新的模型文件夹，例如：
```
outputs/model_v3_20251112_003011/
├── model
├── hyperparameters.json
└── progress.csv
```

#### 步骤 2: 准备 THCHS30 测试集

1. 确认音频文件位置: `/Users/r3ttalynn/Desktop/MT/data_data_thchs30/test/`
2. 准备真实标签文件（从 THCHS30 的转录文本中提取）

#### 步骤 3: 运行评估

```bash
cd test_replication

python evaluate_whisper_plugin.py \
    --whisper_model base \
    --plugin_output_dir outputs/model_v3_20251112_003011 \
    --plugin_model_type baseline \
    --audio_dir /Users/r3ttalynn/Desktop/MT/data_data_thchs30/test/ \
    --ground_truth_file thchs30_test_labels.txt \
    --output results_whisper_plugin.csv \
    --max_samples 10  # 先用 10 个样本测试
```

## 常用命令

### 训练模型
```bash
python train_v3.py
```

### 测试模型（在文本数据上）
```bash
python test.py \
    --model_path outputs/model_v3_YYYYMMDD_HHMMSS/model \
    --hyperparameters outputs/model_v3_YYYYMMDD_HHMMSS/hyperparameters.json \
    --test_data data/test \
    --output results.csv
```

### 使用插件（Python）
```python
from whisper_plugin import WhisperPunctuationPlugin
import whisper

# 初始化
plugin = WhisperPunctuationPlugin(
    output_dir='outputs/model_v3_YYYYMMDD_HHMMSS',
    model_type='baseline'
)
whisper_model = whisper.load_model("base")

# 使用
result = whisper_model.transcribe("audio.wav", language="zh")
text_with_punc = plugin.process(result["text"])
```

### 评估 Whisper + 插件
```bash
python evaluate_whisper_plugin.py \
    --whisper_model base \
    --plugin_output_dir outputs/model_v3_YYYYMMDD_HHMMSS \
    --plugin_model_type baseline \
    --audio_dir /path/to/audio/ \
    --ground_truth_file /path/to/labels.txt \
    --output results.csv
```

## 文件说明

- `train_v3.py` - 训练标点符号恢复模型
- `test.py` - 在文本测试集上评估模型
- `whisper_plugin.py` - Whisper 插件（添加标点）
- `evaluate_whisper_plugin.py` - **评估 Whisper + 插件的端到端性能** ⭐
- `evaluate_with_plots_v1.ipynb` - 生成可视化图表

## 需要帮助？

查看详细文档：
- `WHISPER_USAGE_GUIDE.md` - 完整使用指南
- `WHISPER_PLUGIN_README.md` - 插件 API 文档


