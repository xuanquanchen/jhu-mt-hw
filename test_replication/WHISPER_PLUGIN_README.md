# Whisper 标点符号插件使用指南

## 概述

这是一个用于 Whisper 语音转文字的后处理插件，可以自动为中文转写结果添加标点符号。

## 功能特点

- ✅ 自动为无标点的文本添加标点符号
- ✅ 支持批量处理
- ✅ 支持处理 Whisper 的分段结果
- ✅ 支持基线模型和 Adapter 模型
- ✅ 易于集成到现有项目

## 安装依赖

```bash
pip install torch transformers numpy
```

如果需要与 Whisper 集成：
```bash
pip install openai-whisper
```

## 快速开始

### 1. 保存训练好的模型用于部署

首先，将训练好的模型保存为部署格式：

```bash
python save_model_for_deployment.py \
    --output_dir outputs/model_adapter_YYYYMMDD_HHMMSS \
    --deployment_dir models/punctuation_model \
    --model_type adapter
```

### 2. 基本使用

#### 方式1: 直接使用 PunctuationRestorer

```python
from punctuation_restorer import PunctuationRestorer

# 初始化
restorer = PunctuationRestorer(
    model_path='models/punctuation_model/model.pth',
    hyperparameters_path='models/punctuation_model/hyperparameters.json',
    model_type='adapter'
)

# 处理文本
text = "你好世界今天天气很好"
result = restorer.restore_punctuation(text)
print(result)  # 输出: 你好世界，今天天气很好。
```

#### 方式2: 使用 WhisperPunctuationPlugin

```python
from whisper_plugin import WhisperPunctuationPlugin

# 初始化插件
plugin = WhisperPunctuationPlugin(
    output_dir='models/punctuation_model',
    model_type='adapter'
)

# 处理文本
text = "你好世界今天天气很好"
result = plugin.process(text)
print(result)
```

### 3. 与 Whisper 集成

```python
import whisper
from whisper_plugin import WhisperPunctuationPlugin

# 初始化插件
plugin = WhisperPunctuationPlugin(
    output_dir='models/punctuation_model',
    model_type='adapter'
)

# 加载 Whisper 模型
whisper_model = whisper.load_model("base")

# 转写音频
result = whisper_model.transcribe("audio.mp3", language="zh")

# 处理转写结果
original_text = result["text"]
text_with_punctuation = plugin.process(original_text)

print(f"原始: {original_text}")
print(f"添加标点: {text_with_punctuation}")

# 或者处理分段结果
if "segments" in result:
    processed_segments = plugin.process_segments(result["segments"])
    for segment in processed_segments:
        print(f"{segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")
```

## 命令行使用

```bash
# 测试插件
python whisper_plugin.py \
    --output_dir models/punctuation_model \
    --model_type adapter \
    --text "你好世界今天天气很好"
```

## API 参考

### WhisperPunctuationPlugin

#### `__init__(model_path, hyperparameters_path, output_dir, model_type, device)`

初始化插件。

**参数：**
- `model_path`: 模型权重文件路径（如果提供 `output_dir` 则不需要）
- `hyperparameters_path`: 超参数配置文件路径（如果提供 `output_dir` 则不需要）
- `output_dir`: 模型输出目录（包含 `model` 和 `hyperparameters.json`）
- `model_type`: 模型类型 ('baseline' 或 'adapter')
- `device`: 设备 ('cuda' 或 'cpu')，None 表示自动选择

#### `process(text: str) -> str`

处理单条文本，添加标点符号。

**参数：**
- `text`: Whisper 输出的文本（无标点或标点不完整）

**返回：**
- 添加标点后的文本

#### `process_batch(texts: List[str]) -> List[str]`

批量处理文本。

**参数：**
- `texts`: 文本列表

**返回：**
- 处理后的文本列表

#### `process_segments(segments: List[dict]) -> List[dict]`

处理 Whisper 的分段结果。

**参数：**
- `segments`: Whisper 分段结果列表，每个元素包含 'text' 字段

**返回：**
- 处理后的分段结果，'text' 字段已添加标点

## 模型选择

### 基线模型 (Baseline)
- 全参数微调
- 参数多，性能稳定
- 适合对精度要求高的场景

### Adapter 模型
- 参数少（减少 75.5%）
- 训练速度快
- 显存占用少
- 性能与基线模型相当

**推荐使用 Adapter 模型**，除非有特殊需求。

## 性能优化

1. **使用 GPU**：如果有 GPU，会自动使用，速度提升 5-10 倍
2. **批量处理**：使用 `process_batch` 可以批量处理，提高效率
3. **模型量化**：可以考虑使用模型量化进一步减少显存占用

## 注意事项

1. **首次使用**：首次使用时会自动下载 BERT 模型（bert-base-chinese），需要网络连接
2. **内存要求**：模型较大，建议至少 4GB 可用内存
3. **文本长度**：对于超长文本，会自动分段处理
4. **标点符号**：当前支持：`，`、`。`、`？`（可根据训练数据扩展）

## 故障排除

### 问题1: 模型加载失败
- 检查模型文件路径是否正确
- 确认超参数文件存在
- 检查模型类型是否匹配

### 问题2: 显存不足
- 使用 CPU 模式：`device='cpu'`
- 减小批处理大小
- 使用 Adapter 模型（显存占用更少）

### 问题3: 标点符号不正确
- 检查输入文本是否包含特殊字符
- 确认模型训练时使用的标点符号集
- 尝试预处理文本（移除特殊字符）

## 示例项目结构

```
whisper-punctuation-plugin/
├── models/
│   └── punctuation_model/          # 部署模型目录
│       ├── model.pth
│       ├── hyperparameters.json
│       ├── deployment_config.json
│       ├── model.py
│       ├── model_adapter.py
│       ├── punctuation_restorer.py
│       └── whisper_plugin.py
├── examples/
│   ├── basic_usage.py
│   └── whisper_integration.py
└── README.md
```

## 更新日志

### v1.0.0
- 初始版本
- 支持基线模型和 Adapter 模型
- 基本 API 和 Whisper 集成

## 许可证

请参考项目主 LICENSE 文件。

