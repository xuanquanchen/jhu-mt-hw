# -*- coding: utf-8 -*-
"""
Whisper 插件 - 自动添加标点符号
用于处理 Whisper 语音转文字的输出，自动添加中文标点符号
"""

import os
import sys
from typing import Optional, Union, List
from punctuation_restorer import PunctuationRestorer, create_restorer_from_output_dir


class WhisperPunctuationPlugin:
    """
    Whisper 标点符号插件
    可以作为 Whisper 的后处理步骤，自动为转写结果添加标点符号
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 hyperparameters_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 model_type: str = 'adapter',
                 device: Optional[str] = None):
        """
        初始化插件
        
        Args:
            model_path: 模型权重文件路径（如果提供 output_dir 则不需要）
            hyperparameters_path: 超参数配置文件路径（如果提供 output_dir 则不需要）
            output_dir: 模型输出目录（包含 model 和 hyperparameters.json）
            model_type: 模型类型 ('baseline' 或 'adapter')，默认 'adapter'
            device: 设备 ('cuda' 或 'cpu')，None 表示自动选择
        """
        if output_dir:
            # 从输出目录加载
            self.restorer = create_restorer_from_output_dir(output_dir, model_type)
        elif model_path and hyperparameters_path:
            # 从文件路径加载
            self.restorer = PunctuationRestorer(
                model_path, hyperparameters_path, model_type, device
            )
        else:
            raise ValueError("必须提供 output_dir 或 (model_path, hyperparameters_path)")
        
        print("✓ Whisper Punctuation Plugin initialized")
    
    def process(self, text: str) -> str:
        """
        处理单条文本，添加标点符号
        
        Args:
            text: Whisper 输出的文本（无标点或标点不完整）
        
        Returns:
            添加标点后的文本
        """
        return self.restorer.restore_punctuation(text)
    
    def process_batch(self, texts: Union[List[str], str]) -> Union[List[str], str]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表或单个文本
        
        Returns:
            处理后的文本列表或单个文本
        """
        if isinstance(texts, str):
            return self.process(texts)
        else:
            return self.restorer.restore_batch(texts)
    
    def process_segments(self, segments: List[dict]) -> List[dict]:
        """
        处理 Whisper 的分段结果
        
        Args:
            segments: Whisper 分段结果列表，每个元素包含 'text' 字段
        
        Returns:
            处理后的分段结果，'text' 字段已添加标点
        """
        processed_segments = []
        for segment in segments:
            processed_segment = segment.copy()
            if 'text' in segment:
                processed_segment['text'] = self.process(segment['text'])
            processed_segments.append(processed_segment)
        return processed_segments


# ============================================================================
# Whisper 集成示例
# ============================================================================

def integrate_with_whisper_example():
    """
    与 Whisper 集成的示例代码
    """
    try:
        import whisper
    except ImportError:
        print("请先安装 whisper: pip install openai-whisper")
        return
    
    # 1. 初始化插件（使用训练好的模型）
    plugin = WhisperPunctuationPlugin(
        output_dir='outputs/model_adapter_YYYYMMDD_HHMMSS',  # 替换为实际的输出目录
        model_type='adapter'
    )
    
    # 2. 加载 Whisper 模型
    whisper_model = whisper.load_model("base")
    
    # 3. 转写音频
    audio_file = "audio.mp3"  # 替换为实际的音频文件
    result = whisper_model.transcribe(audio_file, language="zh")
    
    # 4. 处理转写结果
    # 方式1: 处理整个文本
    original_text = result["text"]
    text_with_punctuation = plugin.process(original_text)
    print(f"原始文本: {original_text}")
    print(f"添加标点: {text_with_punctuation}")
    
    # 方式2: 处理分段结果
    if "segments" in result:
        processed_segments = plugin.process_segments(result["segments"])
        for segment in processed_segments:
            print(f"时间: {segment['start']:.2f}s - {segment['end']:.2f}s")
            print(f"文本: {segment['text']}")
            print()


def standalone_usage_example():
    """
    独立使用示例（不依赖 Whisper）
    """
    # 初始化插件
    plugin = WhisperPunctuationPlugin(
        output_dir='outputs/model_adapter_YYYYMMDD_HHMMSS',  # 替换为实际的输出目录
        model_type='adapter'
    )
    
    # 处理文本
    texts = [
        "你好世界今天天气很好",
        "这是Whisper输出的无标点文本我们需要添加标点符号",
        "人工智能是未来的发展方向",
    ]
    
    print("处理结果:")
    for text in texts:
        result = plugin.process(text)
        print(f"输入: {text}")
        print(f"输出: {result}")
        print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Whisper 标点符号插件')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='模型输出目录')
    parser.add_argument('--model_type', type=str, default='adapter',
                        choices=['baseline', 'adapter'],
                        help='模型类型')
    parser.add_argument('--text', type=str, default=None,
                        help='测试文本（如果不提供，则运行示例）')
    
    args = parser.parse_args()
    
    # 初始化插件
    plugin = WhisperPunctuationPlugin(
        output_dir=args.output_dir,
        model_type=args.model_type
    )
    
    if args.text:
        # 处理单个文本
        result = plugin.process(args.text)
        print(f"输入: {args.text}")
        print(f"输出: {result}")
    else:
        # 运行示例
        print("运行独立使用示例...")
        standalone_usage_example()

