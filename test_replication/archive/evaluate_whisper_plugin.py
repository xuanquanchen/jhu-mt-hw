# -*- coding: utf-8 -*-
"""
评估 Whisper + 标点符号插件的端到端性能
用于评估在 THCHS30 数据集上的表现

使用方法:
python evaluate_whisper_plugin.py \
    --whisper_model base \
    --plugin_output_dir outputs/model_v3_YYYYMMDD_HHMMSS \
    --audio_dir /Users/r3ttalynn/Desktop/MT/data_data_thchs30/ \
    --ground_truth_file /path/to/ground_truth.txt \
    --output results_whisper_plugin.csv
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re

try:
    import whisper
except ImportError:
    print("请先安装 whisper: pip install openai-whisper")
    exit(1)

from whisper_plugin import WhisperPunctuationPlugin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_ground_truth(ground_truth_file):
    """
    加载真实标签文件
    
    支持格式:
    1. 每行一个句子，带标点: "你好世界，今天天气很好。"
    2. 每行 word punctuation: "你好 O\n世界 ，\n今天 O\n..."
    """
    ground_truth = []
    
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 尝试检测格式
    first_line = lines[0].strip() if lines else ""
    
    if '\t' in first_line or ' ' in first_line and len(first_line.split()) == 2:
        # 格式2: word punctuation
        current_sentence = []
        for line in lines:
            line = line.strip()
            if not line:
                if current_sentence:
                    # 合并成句子
                    sentence = ''.join([word + (punc if punc != 'O' else '') 
                                       for word, punc in current_sentence])
                    ground_truth.append(sentence)
                    current_sentence = []
                continue
            
            parts = line.split()
            if len(parts) == 2:
                word, punc = parts
                current_sentence.append((word, punc))
        
        if current_sentence:
            sentence = ''.join([word + (punc if punc != 'O' else '') 
                               for word, punc in current_sentence])
            ground_truth.append(sentence)
    else:
        # 格式1: 每行一个句子
        ground_truth = [line.strip() for line in lines if line.strip()]
    
    return ground_truth


def find_audio_files(audio_dir):
    """在目录中查找所有音频文件"""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        print(f"警告: 音频目录不存在: {audio_dir}")
        return audio_files
    
    for ext in audio_extensions:
        audio_files.extend(audio_path.rglob(f'*{ext}'))
    
    return sorted(audio_files)


def remove_punctuation(text):
    """移除文本中的标点符号，用于比较"""
    # 移除所有中文标点
    punc_pattern = r'[，。？！；：、""''（）【】《》〈〉『』「」]'
    text_no_punc = re.sub(punc_pattern, '', text)
    # 移除空格
    text_no_punc = re.sub(r'\s+', '', text_no_punc)
    return text_no_punc


def calculate_punctuation_accuracy(pred_text, true_text, supported_punctuation=None):
    """
    计算标点符号准确率
    
    比较预测文本和真实文本中的标点符号位置和类型
    
    Args:
        pred_text: 预测文本
        true_text: 真实文本
        supported_punctuation: 模型支持的标点符号列表，例如 ['，', '。', '？']
                             如果为 None，则使用默认的宽泛模式
    """
    if supported_punctuation is None:
        # 默认：使用宽泛的标点符号模式（向后兼容）
        punc_pattern = r'([，。？！；：、""''（）【】《》〈〉『』「」])'
    else:
        # 只匹配模型支持的标点符号
        # 转义特殊字符用于正则表达式
        escaped_puncs = [re.escape(p) for p in supported_punctuation if p != 'O']
        if escaped_puncs:
            punc_pattern = r'([' + ''.join(escaped_puncs) + r'])'
        else:
            punc_pattern = r'(?!)'  # 不匹配任何内容
    
    pred_puncs = [(m.start(), m.group(1)) for m in re.finditer(punc_pattern, pred_text)]
    true_puncs = [(m.start(), m.group(1)) for m in re.finditer(punc_pattern, true_text)]
    
    if len(true_puncs) == 0:
        # 如果真实文本没有标点，检查预测文本是否也没有
        # 返回字典格式以保持一致性
        if len(pred_puncs) == 0:
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'correct': 0,
                'pred_count': 0,
                'true_count': 0
            }
        else:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'correct': 0,
                'pred_count': len(pred_puncs),
                'true_count': 0
            }
    
    # 计算准确率
    # 方法1: 精确匹配（位置和类型都正确）
    correct = 0
    true_positions = {pos: punc for pos, punc in true_puncs}
    pred_positions = {pos: punc for pos, punc in pred_puncs}
    
    for pos, punc in true_puncs:
        if pos in pred_positions and pred_positions[pos] == punc:
            correct += 1
    
    precision = correct / len(pred_puncs) if len(pred_puncs) > 0 else 0.0
    recall = correct / len(true_puncs) if len(true_puncs) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': correct,
        'pred_count': len(pred_puncs),
        'true_count': len(true_puncs)
    }


def evaluate_whisper_plugin(whisper_model_name, plugin, audio_files, ground_truth, 
                            max_samples=None, language='zh', supported_punctuation=None):
    """
    评估 Whisper + 插件的端到端性能
    
    Args:
        whisper_model_name: Whisper 模型名称 (tiny, base, small, medium, large)
        plugin: WhisperPunctuationPlugin 实例
        audio_files: 音频文件列表
        ground_truth: 真实标签列表
        max_samples: 最大评估样本数（None 表示全部）
        language: 语言代码（'zh' 表示中文）
    """
    print(f"加载 Whisper 模型: {whisper_model_name}")
    whisper_model = whisper.load_model(whisper_model_name)
    print("✓ Whisper 模型加载完成")
    
    # 限制样本数
    if max_samples:
        audio_files = audio_files[:max_samples]
        ground_truth = ground_truth[:max_samples]
    
    if len(audio_files) != len(ground_truth):
        print(f"警告: 音频文件数量 ({len(audio_files)}) 与真实标签数量 ({len(ground_truth)}) 不匹配")
        min_len = min(len(audio_files), len(ground_truth))
        audio_files = audio_files[:min_len]
        ground_truth = ground_truth[:min_len]
    
    results = []
    
    print(f"\n开始评估 {len(audio_files)} 个样本...")
    
    for i, (audio_file, true_text) in enumerate(tqdm(zip(audio_files, ground_truth), 
                                                      total=len(audio_files),
                                                      desc="处理音频")):
        try:
            # 1. Whisper 转写
            result = whisper_model.transcribe(str(audio_file), language=language)
            whisper_text = result["text"].strip()
            
            # 2. 插件添加标点
            text_with_punctuation = plugin.process(whisper_text)
            
            # 3. 计算指标
            # 3.1 文本相似度（移除标点后）
            whisper_no_punc = remove_punctuation(whisper_text)
            true_no_punc = remove_punctuation(true_text)
            text_accuracy = 1.0 if whisper_no_punc == true_no_punc else 0.0
            
            # 3.2 标点符号准确率（只评估模型支持的标点）
            punc_metrics = calculate_punctuation_accuracy(text_with_punctuation, true_text, supported_punctuation)
            
            results.append({
                'audio_file': str(audio_file),
                'whisper_text': whisper_text,
                'text_with_punctuation': text_with_punctuation,
                'ground_truth': true_text,
                'text_accuracy': text_accuracy,
                'punc_precision': punc_metrics['precision'],
                'punc_recall': punc_metrics['recall'],
                'punc_f1': punc_metrics['f1'],
                'punc_correct': punc_metrics['correct'],
                'punc_pred_count': punc_metrics['pred_count'],
                'punc_true_count': punc_metrics['true_count'],
            })
            
        except Exception as e:
            print(f"\n错误处理文件 {audio_file}: {e}")
            results.append({
                'audio_file': str(audio_file),
                'whisper_text': '',
                'text_with_punctuation': '',
                'ground_truth': true_text,
                'text_accuracy': 0.0,
                'punc_precision': 0.0,
                'punc_recall': 0.0,
                'punc_f1': 0.0,
                'punc_correct': 0,
                'punc_pred_count': 0,
                'punc_true_count': len(re.findall(r'[，。？！；：、""''（）【】《》]', true_text)),
            })
    
    return pd.DataFrame(results)


def print_summary(results_df):
    """打印评估结果摘要"""
    print("\n" + "=" * 80)
    print("评估结果摘要")
    print("=" * 80)
    
    # 文本准确率
    text_acc = results_df['text_accuracy'].mean()
    print(f"\n文本准确率 (移除标点后): {text_acc:.4f} ({text_acc*100:.2f}%)")
    
    # 标点符号指标
    punc_precision = results_df['punc_precision'].mean()
    punc_recall = results_df['punc_recall'].mean()
    punc_f1 = results_df['punc_f1'].mean()
    
    print(f"\n标点符号指标:")
    print(f"  Precision: {punc_precision:.4f} ({punc_precision*100:.2f}%)")
    print(f"  Recall:    {punc_recall:.4f} ({punc_recall*100:.2f}%)")
    print(f"  F1-Score:  {punc_f1:.4f} ({punc_f1*100:.2f}%)")
    
    # 标点符号数量统计
    total_pred = results_df['punc_pred_count'].sum()
    total_true = results_df['punc_true_count'].sum()
    total_correct = results_df['punc_correct'].sum()
    
    print(f"\n标点符号数量:")
    print(f"  预测总数: {total_pred}")
    print(f"  真实总数: {total_true}")
    print(f"  正确数量: {total_correct}")
    
    # 样本统计
    print(f"\n样本统计:")
    print(f"  总样本数: {len(results_df)}")
    print(f"  成功处理: {len(results_df[results_df['text_accuracy'] >= 0])}")
    
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估 Whisper + 标点符号插件')
    parser.add_argument('--whisper_model', type=str, default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper 模型大小')
    parser.add_argument('--plugin_output_dir', type=str, required=True,
                       help='插件模型输出目录 (outputs/model_v3_YYYYMMDD_HHMMSS)')
    parser.add_argument('--plugin_model_type', type=str, default='baseline',
                       choices=['baseline', 'adapter'],
                       help='插件模型类型')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='音频文件目录')
    parser.add_argument('--ground_truth_file', type=str, required=True,
                       help='真实标签文件路径')
    parser.add_argument('--output', type=str, default='results_whisper_plugin.csv',
                       help='结果输出文件路径')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大评估样本数（用于快速测试）')
    parser.add_argument('--language', type=str, default='zh',
                       help='语言代码（默认: zh 中文）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Whisper + 标点符号插件评估")
    print("=" * 80)
    print(f"Whisper 模型: {args.whisper_model}")
    print(f"插件模型目录: {args.plugin_output_dir}")
    print(f"插件模型类型: {args.plugin_model_type}")
    print(f"音频目录: {args.audio_dir}")
    print(f"真实标签文件: {args.ground_truth_file}")
    print("=" * 80)
    
    # 1. 加载插件并读取支持的标点符号
    print("\n加载标点符号插件...")
    try:
        plugin = WhisperPunctuationPlugin(
            output_dir=args.plugin_output_dir,
            model_type=args.plugin_model_type
        )
        print("✓ 插件加载完成")
        
        # 读取模型支持的标点符号
        hyperparameters_path = os.path.join(args.plugin_output_dir, 'hyperparameters.json')
        if os.path.exists(hyperparameters_path):
            with open(hyperparameters_path, 'r', encoding='utf-8') as f:
                hyperparams = json.load(f)
            supported_punctuation = list(hyperparams.get('punctuation_enc', {}).keys())
            # 移除 'O'（无标点）
            supported_punctuation = [p for p in supported_punctuation if p != 'O']
            print(f"✓ 模型支持的标点符号: {', '.join(supported_punctuation)}")
        else:
            print("⚠ 警告: 未找到 hyperparameters.json，将使用默认标点符号模式")
            supported_punctuation = None
    except Exception as e:
        print(f"✗ 插件加载失败: {e}")
        exit(1)
    
    # 2. 加载真实标签
    print(f"\n加载真实标签: {args.ground_truth_file}")
    try:
        ground_truth = load_ground_truth(args.ground_truth_file)
        print(f"✓ 加载了 {len(ground_truth)} 个真实标签")
    except Exception as e:
        print(f"✗ 加载真实标签失败: {e}")
        exit(1)
    
    # 3. 查找音频文件
    print(f"\n查找音频文件: {args.audio_dir}")
    audio_files = find_audio_files(args.audio_dir)
    print(f"✓ 找到 {len(audio_files)} 个音频文件")
    
    if len(audio_files) == 0:
        print("错误: 未找到音频文件")
        exit(1)
    
    # 4. 评估
    print("\n" + "=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    results_df = evaluate_whisper_plugin(
        args.whisper_model,
        plugin,
        audio_files,
        ground_truth,
        max_samples=args.max_samples,
        language=args.language,
        supported_punctuation=supported_punctuation
    )
    
    # 5. 打印结果
    print_summary(results_df)
    
    # 6. 保存结果
    results_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存到: {args.output}")
    
    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)

