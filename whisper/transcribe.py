#!/usr/bin/env python3
"""
Whisper transcription script
Usage: python transcribe.py --file "path/to/audio/file"
"""

import argparse
import os
import sys
import torch
import whisper
from whisper.tokenizer import get_tokenizer

# Try to import opencc for traditional to simplified Chinese conversion
try:
    import opencc
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False
    print("Warning: opencc not installed. Install with: pip install opencc-python-reimplemented")

# Manual traditional to simplified Chinese conversion for common characters
TRADITIONAL_TO_SIMPLIFIED = {
    '中國': '中国', '保護': '保护', '野生動物': '野生动物', '法律法規': '法律法规',
    '體系': '体系', '建設': '建设', '出臺': '出台', '中華人民共和國': '中华人民共和国',
    '修訂': '修订', '調整': '调整', '發布': '发布', '國家': '国家', '重點': '重点',
    '明路': '名录', '生態': '生态', '科學': '科学', '社會': '社会', '價值': '价值',
    '鳥類': '鸟类', '達到': '达到', '種': '种', '擴大': '扩大', '近': '近',
    '始終': '始终', '重視': '重视', '加強': '加强', '內機': '内机', '機器': '机器',
    '電腦': '电脑', '間': '间', '進一步': '进一步', '完善': '完善', '相關': '相关',
    '施地': '湿地', '野生': '野生', '動物': '动物', '有': '有', '重要': '重要',
    '路生': '陆生', '數': '数', '15%': '15%',
    # Individual character conversions
    '終': '终', '重': '重', '視': '视', '加': '加', '強': '强', '內': '内', '機': '机',
    '始': '始', '了': '了', '系': '系', '电': '电', '脑': '脑', '保': '保', '护': '护',
    '间': '间', '进': '进', '一': '一', '步': '步', '完': '完', '善': '善', '相': '相',
    '关': '关', '法': '法', '律': '律', '规': '规', '体': '体', '系': '系', '建': '建',
    '设': '设', '出': '出', '台': '台', '中': '中', '华': '华', '人': '人', '民': '民',
    '共': '共', '和': '和', '国': '国', '湿': '湿', '地': '地', '修': '修', '订': '订',
    '野': '野', '生': '生', '动': '动', '物': '物', '调': '调', '整': '整', '发': '发',
    '布': '布', '国': '国', '家': '家', '重': '重', '点': '点', '名': '名', '录': '录',
    '有': '有', '重': '重', '要': '要', '生': '生', '态': '态', '科': '科', '学': '学',
    '社': '社', '会': '会', '价': '价', '值': '值', '陆': '陆', '鸟': '鸟', '类': '类',
    '达': '达', '到': '到', '种': '种', '扩': '扩', '大': '大', '近': '近'
}

def convert_traditional_to_simplified(text):
    """Convert traditional Chinese characters to simplified Chinese"""
    result = text
    for traditional, simplified in TRADITIONAL_TO_SIMPLIFIED.items():
        result = result.replace(traditional, simplified)
    return result

def transcribe_audio(audio_path, model_name="tiny", language=None, temperature=0.0, translate=False):
    """Transcribe audio file using Whisper"""
    
    task_type = "Translating to Chinese" if translate else "Transcribing"
    print(f"{task_type}: {audio_path}")
    print(f"Model: {model_name}")
    print(f"Language: {language or 'auto-detect'}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return None
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading Whisper model...")
    try:
        model = whisper.load_model(model_name).to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Transcribe or translate audio
    action = "Translating audio to Chinese..." if translate else "Transcribing audio..."
    print(action)
    try:
        result = model.transcribe(
            audio_path, 
            language=language, 
            task="translate" if translate else "transcribe",
            temperature=temperature, 
            word_timestamps=True,
            verbose=False
        )
        
        # Convert to simplified Chinese if translating OR if transcribing Chinese
        if translate or (language == "zh"):
            if OPENCC_AVAILABLE:
                converter = opencc.OpenCC('t2s')  # Traditional to Simplified
                result['text'] = converter.convert(result['text'])
                # Also convert segment texts and word-level timestamps
                for segment in result.get('segments', []):
                    segment['text'] = converter.convert(segment['text'])
                    # Convert word-level timestamps
                    if 'words' in segment:
                        for word_info in segment['words']:
                            word_info['word'] = converter.convert(word_info['word'])
            else:
                # Use manual conversion as fallback
                result['text'] = convert_traditional_to_simplified(result['text'])
                # Also convert segment texts and word-level timestamps
                for segment in result.get('segments', []):
                    segment['text'] = convert_traditional_to_simplified(segment['text'])
                    # Convert word-level timestamps
                    if 'words' in segment:
                        for word_info in segment['words']:
                            word_info['word'] = convert_traditional_to_simplified(word_info['word'])
        
        # Display results
        print("\n" + "=" * 60)
        result_type = "TRANSLATION RESULTS" if translate else "TRANSCRIPTION RESULTS"
        print(result_type)
        print("=" * 60)
        print(f"Detected language: {result['language']}")
        print(f"Full text: {result['text']}")
        
        # Show segments with timestamps
        if 'segments' in result and result['segments']:
            print(f"\nSegments ({len(result['segments'])} total):")
            for i, segment in enumerate(result['segments'], 1):
                print(f"  {i:2d}. [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
        
        # Show word-level timestamps (first few words)
        if 'segments' in result and result['segments']:
            print(f"\nWord-level timestamps (first segment):")
            first_segment = result['segments'][0]
            if 'words' in first_segment and first_segment['words']:
                for word_info in first_segment['words'][:10]:  # Show first 10 words
                    print(f"  [{word_info['start']:.2f}s - {word_info['end']:.2f}s] '{word_info['word']}'")
                if len(first_segment['words']) > 10:
                    print(f"  ... and {len(first_segment['words']) - 10} more words")
        
        return result
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper")
    parser.add_argument("--file", "-f", required=True, help="Path to audio file")
    parser.add_argument("--model", "-m", default="tiny", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: tiny)")
    parser.add_argument("--en", action="store_true", help="Transcribe in English")
    parser.add_argument("--cn", action="store_true", help="Transcribe in Chinese")
    parser.add_argument("--translate", action="store_true", help="Translate to Simplified Chinese")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                       help="Temperature for sampling (default: 0.0)")
    
    args = parser.parse_args()
    
    # Determine language based on flags
    language = None
    if args.en and args.cn:
        print("Error: Cannot specify both --en and --cn. Choose one.")
        sys.exit(1)
    elif args.en:
        language = "en"
    elif args.cn:
        language = "zh"
    
    # Check for conflicting options
    if args.translate and (args.en or args.cn):
        print("Error: --translate cannot be used with --en or --cn. Use --translate alone.")
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    # Transcribe or translate
    result = transcribe_audio(
        audio_path=args.file,
        model_name=args.model,
        language=language,
        temperature=args.temperature,
        translate=args.translate
    )
    
    if result:
        action = "Translation" if args.translate else "Transcription"
        print(f"\n{action} completed successfully!")
    else:
        action = "Translation" if args.translate else "Transcription"
        print(f"\n{action} failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
