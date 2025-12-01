# -*- coding: utf-8 -*-
"""
从 THCHS30 数据集中提取真实标签文件
用于 Whisper + 插件评估
"""

import os
import argparse
from pathlib import Path


def extract_labels_from_trn(trn_file):
    """
    从 .trn 文件中提取文本标签
    
    THCHS30 的 .trn 文件格式通常是：
    第一行：汉字（分词，可能无标点）
    第二行：拼音
    第三行：音素
    
    注意：THCHS30 的原始转录可能没有标点，我们需要使用第一行的汉字文本
    """
    with open(trn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 第一行是汉字文本（分词，可能无标点）
    if len(lines) >= 1:
        text = lines[0].strip()
        # 移除空格，合并成连续文本（因为原始数据是分词的）
        text = text.replace(' ', '')
        return text
    else:
        return ""


def prepare_labels(audio_dir, output_file):
    """
    从 THCHS30 测试集中提取所有标签
    
    Args:
        audio_dir: 音频文件目录（包含 .wav 和 .trn 文件）
        output_file: 输出标签文件路径
    """
    audio_path = Path(audio_dir)
    
    # 找到所有 .wav 文件
    wav_files = sorted(audio_path.glob('*.wav'))
    
    print(f"找到 {len(wav_files)} 个音频文件")
    
    labels = []
    missing_trn = []
    
    for wav_file in wav_files:
        # 先检查 test 目录下的 .trn 文件（可能是符号链接或路径引用）
        trn_file = wav_file.with_suffix('.wav.trn')
        
        if not trn_file.exists():
            trn_file = wav_file.with_suffix('.trn')
        
        # 如果 test 目录下的 .trn 文件存在但只包含路径，读取实际文件
        if trn_file.exists():
            with open(trn_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 如果内容是路径，读取实际文件
            if content.startswith('../') or content.startswith('./'):
                actual_path = audio_path.parent / content
                if actual_path.exists():
                    label = extract_labels_from_trn(actual_path)
                else:
                    # 尝试在 data 目录下查找
                    actual_path = audio_path.parent / 'data' / wav_file.name.replace('.wav', '.wav.trn')
                    if actual_path.exists():
                        label = extract_labels_from_trn(actual_path)
                    else:
                        label = ""
            else:
                # 直接读取内容
                label = extract_labels_from_trn(trn_file)
        else:
            # 尝试在 data 目录下查找
            actual_path = audio_path.parent / 'data' / wav_file.name.replace('.wav', '.wav.trn')
            if actual_path.exists():
                label = extract_labels_from_trn(actual_path)
            else:
                label = ""
                missing_trn.append(wav_file.name)
        
        if label:
            labels.append(label)
        else:
            if wav_file.name not in missing_trn:
                print(f"警告: {wav_file.name} 的转录为空")
            labels.append("")  # 保留空行以保持顺序
    
    # 保存标签文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + '\n')
    
    print(f"\n✓ 已提取 {len([l for l in labels if l])} 个标签")
    print(f"✓ 保存到: {output_file}")
    
    if missing_trn:
        print(f"\n警告: {len(missing_trn)} 个文件缺少转录文件")
        if len(missing_trn) <= 10:
            print("缺少转录的文件:")
            for f in missing_trn:
                print(f"  - {f}")
    
    return len([l for l in labels if l])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从 THCHS30 提取真实标签')
    parser.add_argument('--audio_dir', type=str, 
                       default='/Users/r3ttalynn/Desktop/MT/data_thchs30/test/',
                       help='THCHS30 测试集目录')
    parser.add_argument('--output', type=str,
                       default='thchs30_test_labels.txt',
                       help='输出标签文件路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("THCHS30 标签提取工具")
    print("=" * 60)
    print(f"音频目录: {args.audio_dir}")
    print(f"输出文件: {args.output}")
    print("=" * 60)
    
    if not os.path.exists(args.audio_dir):
        print(f"错误: 目录不存在: {args.audio_dir}")
        exit(1)
    
    count = prepare_labels(args.audio_dir, args.output)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

