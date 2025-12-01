# -*- coding: utf-8 -*-
"""
Whisper 插件使用示例
演示如何将标点符号恢复器集成到 Whisper 工作流中
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

from whisper_plugin import WhisperPunctuationPlugin


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)
    
    # 初始化插件（需要替换为实际的模型路径）
    # 训练完成后，可以从 outputs/ 目录中找到模型
    plugin = WhisperPunctuationPlugin(
        output_dir='outputs/model_adapter_YYYYMMDD_HHMMSS',  # 替换为实际路径
        model_type='adapter'
    )
    
    # 模拟 Whisper 输出的无标点文本
    whisper_outputs = [
        "你好世界今天天气很好",
        "人工智能是未来的发展方向我们需要不断学习",
        "这个项目非常有趣我很喜欢",
    ]
    
    print("\n处理结果:")
    for text in whisper_outputs:
        result = plugin.process(text)
        print(f"输入: {text}")
        print(f"输出: {result}\n")


def example_2_batch_processing():
    """示例2: 批量处理"""
    print("=" * 60)
    print("示例2: 批量处理")
    print("=" * 60)
    
    plugin = WhisperPunctuationPlugin(
        output_dir='outputs/model_adapter_YYYYMMDD_HHMMSS',
        model_type='adapter'
    )
    
    texts = [
        "第一段文本没有标点符号",
        "第二段文本也需要添加标点",
        "第三段文本同样需要处理",
    ]
    
    results = plugin.process_batch(texts)
    
    print("\n批量处理结果:")
    for original, processed in zip(texts, results):
        print(f"原始: {original}")
        print(f"处理: {processed}\n")


def example_3_whisper_integration():
    """示例3: 与 Whisper 集成"""
    print("=" * 60)
    print("示例3: 与 Whisper 集成")
    print("=" * 60)
    
    try:
        import whisper
    except ImportError:
        print("请先安装 whisper: pip install openai-whisper")
        return
    
    # 初始化插件
    plugin = WhisperPunctuationPlugin(
        output_dir='outputs/model_adapter_YYYYMMDD_HHMMSS',
        model_type='adapter'
    )
    
    # 加载 Whisper 模型
    print("加载 Whisper 模型...")
    whisper_model = whisper.load_model("base")
    
    # 转写音频（需要实际的音频文件）
    audio_file = "audio.mp3"  # 替换为实际的音频文件路径
    
    if not os.path.exists(audio_file):
        print(f"音频文件不存在: {audio_file}")
        print("跳过实际转写，使用模拟数据...")
        
        # 模拟 Whisper 输出
        mock_result = {
            "text": "你好世界今天天气很好",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "你好世界"},
                {"start": 2.5, "end": 5.0, "text": "今天天气很好"},
            ]
        }
    else:
        print(f"转写音频: {audio_file}")
        mock_result = whisper_model.transcribe(audio_file, language="zh")
    
    # 处理整个文本
    print("\n处理整个文本:")
    original_text = mock_result["text"]
    text_with_punctuation = plugin.process(original_text)
    print(f"原始: {original_text}")
    print(f"添加标点: {text_with_punctuation}")
    
    # 处理分段结果
    if "segments" in mock_result:
        print("\n处理分段结果:")
        processed_segments = plugin.process_segments(mock_result["segments"])
        for segment in processed_segments:
            print(f"时间: {segment['start']:.2f}s - {segment['end']:.2f}s")
            print(f"文本: {segment['text']}\n")


def example_4_real_time_processing():
    """示例4: 实时处理（模拟）"""
    print("=" * 60)
    print("示例4: 实时处理模拟")
    print("=" * 60)
    
    plugin = WhisperPunctuationPlugin(
        output_dir='outputs/model_adapter_YYYYMMDD_HHMMSS',
        model_type='adapter'
    )
    
    # 模拟实时接收的文本片段
    text_chunks = [
        "第一句话",
        "第二句话",
        "第三句话",
    ]
    
    print("实时处理文本片段:")
    for i, chunk in enumerate(text_chunks, 1):
        processed = plugin.process(chunk)
        print(f"片段 {i}: {chunk} -> {processed}")
    
    # 或者累积处理
    print("\n累积处理:")
    accumulated_text = ""
    for chunk in text_chunks:
        accumulated_text += chunk
        processed = plugin.process(accumulated_text)
        print(f"累积: {accumulated_text} -> {processed}")


if __name__ == '__main__':
    import sys
    
    print("\n" + "=" * 60)
    print("Whisper 标点符号插件 - 使用示例")
    print("=" * 60)
    print("\n注意: 请先训练模型或使用已训练的模型")
    print("将示例中的 'outputs/model_adapter_YYYYMMDD_HHMMSS' 替换为实际的模型路径")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        if example_num == 1:
            example_1_basic_usage()
        elif example_num == 2:
            example_2_batch_processing()
        elif example_num == 3:
            example_3_whisper_integration()
        elif example_num == 4:
            example_4_real_time_processing()
        else:
            print(f"未知示例编号: {example_num}")
    else:
        print("\n运行所有示例...")
        print("\n提示: 使用 'python example_whisper_usage.py <编号>' 运行特定示例")
        print("  1 - 基本使用")
        print("  2 - 批量处理")
        print("  3 - Whisper 集成")
        print("  4 - 实时处理模拟")
        print("\n由于需要实际的模型文件，这里只显示示例代码结构。")

