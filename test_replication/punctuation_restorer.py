# -*- coding: utf-8 -*-
"""
标点符号恢复器 - 用于 Whisper 插件
处理无标点的文本，自动添加标点符号
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
import numpy as np
from transformers import BertTokenizer
import json
import re

from model import BertChineseEmbSlimCNNlstmBert
from model_adapter import BertChineseEmbSlimCNNlstmBertAdapter


class PunctuationRestorer:
    """
    标点符号恢复器类
    用于处理 Whisper 输出的无标点文本，自动添加标点符号
    """
    
    def __init__(self, model_path, hyperparameters_path, model_type='baseline', device=None):
        """
        初始化标点符号恢复器
        
        Args:
            model_path: 模型权重文件路径
            hyperparameters_path: 超参数配置文件路径
            model_type: 模型类型 ('baseline' 或 'adapter')
            device: 设备 ('cuda' 或 'cpu')，None 表示自动选择
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on device: {self.device}")
        
        # 加载超参数
        with open(hyperparameters_path, 'r', encoding='utf-8') as f:
            self.hyperparams = json.load(f)
        
        self.seq_len = self.hyperparams['seq_len']
        self.dropout = self.hyperparams['dropout']
        self.punctuation_enc = self.hyperparams['punctuation_enc']
        self.output_size = len(self.punctuation_enc)
        self.model_type = model_type
        
        # 创建反向映射
        self.id_to_punctuation = {v: k for k, v in self.punctuation_enc.items()}
        
        # 加载 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', do_lower_case=True)
        
        # 加载模型
        if model_type == 'adapter':
            use_adapter = self.hyperparams.get('use_adapter', True)
            adapter_size = self.hyperparams.get('adapter_size', 384)
            use_attention_fusion = self.hyperparams.get('use_attention_fusion', True)
            
            self.model = BertChineseEmbSlimCNNlstmBertAdapter(
                self.seq_len, self.output_size, self.dropout, None,
                use_adapter=use_adapter,
                adapter_size=adapter_size,
                use_attention_fusion=use_attention_fusion
            )
        else:
            self.model = BertChineseEmbSlimCNNlstmBert(
                self.seq_len, self.output_size, self.dropout, None
            )
        
        # 加载权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully ({model_type})")
    
    def preprocess_text(self, text):
        """
        预处理文本：移除现有标点，分词
        
        Args:
            text: 输入文本（可能包含或不包含标点）
        
        Returns:
            words: 词列表
        """
        # 移除所有标点符号
        punctuation_chars = '，。？！；：、""''（）【】《》'
        text_clean = text.translate(str.maketrans('', '', punctuation_chars))
        
        # 按字符分割（中文按字符，英文按单词）
        words = []
        current_word = ''
        
        for char in text_clean:
            if char.strip() == '':
                if current_word:
                    words.append(current_word)
                    current_word = ''
            elif '\u4e00' <= char <= '\u9fff':  # 中文字符
                words.append(char)
            else:  # 英文或其他字符
                current_word += char
        
        if current_word:
            words.append(current_word)
        
        return [w for w in words if w.strip()]
    
    def encode_text(self, words):
        """
        将词列表编码为 token ids
        
        Args:
            words: 词列表
        
        Returns:
            token_ids: token id 列表
            word_to_tokens: 词到 token 的映射
        """
        token_ids = []
        word_to_tokens = []
        
        for word in words:
            tokens = self.tokenizer.tokenize(word)
            word_token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            if len(word_token_ids) > 0:
                token_ids.extend(word_token_ids)
                word_to_tokens.append((len(token_ids) - len(word_token_ids), len(token_ids)))
        
        return token_ids, word_to_tokens
    
    def predict_punctuation(self, token_ids):
        """
        预测标点符号
        
        Args:
            token_ids: token id 列表
        
        Returns:
            predictions: 标点符号预测列表（每个 token 对应一个）
        """
        # 如果文本太短，直接返回
        if len(token_ids) == 0:
            return []
        
        # 如果文本太长，需要分段处理
        if len(token_ids) > self.seq_len:
            # 分段处理
            all_predictions = []
            for i in range(0, len(token_ids), self.seq_len):
                segment = token_ids[i:i + self.seq_len]
                segment_predictions = self._predict_segment(segment)
                all_predictions.extend(segment_predictions)
            return all_predictions
        else:
            return self._predict_segment(token_ids)
    
    def _predict_segment(self, token_ids):
        """
        预测一个固定长度的片段
        
        Args:
            token_ids: token id 列表（长度 <= seq_len）
        
        Returns:
            predictions: 标点符号预测列表
        """
        # 填充到 seq_len
        padded = token_ids + [0] * (self.seq_len - len(token_ids))
        input_tensor = torch.tensor([padded], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)  # [1 * seq_len, num_classes]
            predictions = output.argmax(dim=1).cpu().numpy()
        
        # 只返回实际长度的预测
        return predictions[:len(token_ids)].tolist()
    
    def restore_punctuation(self, text):
        """
        恢复文本的标点符号（主接口）
        
        Args:
            text: 输入文本（无标点或已有标点）
        
        Returns:
            text_with_punctuation: 添加标点后的文本
        """
        # 预处理：移除现有标点，分词
        words = self.preprocess_text(text)
        
        if len(words) == 0:
            return text
        
        # 编码
        token_ids, word_to_tokens = self.encode_text(words)
        
        if len(token_ids) == 0:
            return text
        
        # 预测标点
        token_predictions = self.predict_punctuation(token_ids)
        
        # 将 token 级别的预测映射到词级别
        # 对于每个词，使用最后一个 token 的预测作为该词的标点
        word_punctuations = []
        for start_idx, end_idx in word_to_tokens:
            # 使用最后一个 token 的预测
            if end_idx > 0:
                word_punctuation = token_predictions[end_idx - 1]
            else:
                word_punctuation = 0  # 'O'
            word_punctuations.append(word_punctuation)
        
        # 构建带标点的文本
        result = []
        for word, punc_id in zip(words, word_punctuations):
            result.append(word)
            punctuation = self.id_to_punctuation.get(punc_id, '')
            if punctuation and punctuation != 'O':
                result.append(punctuation)
        
        return ''.join(result)
    
    def restore_batch(self, texts):
        """
        批量恢复标点符号
        
        Args:
            texts: 文本列表
        
        Returns:
            results: 恢复标点后的文本列表
        """
        return [self.restore_punctuation(text) for text in texts]


def create_restorer_from_output_dir(output_dir, model_type='baseline'):
    """
    从输出目录创建恢复器（便捷函数）
    
    Args:
        output_dir: 模型输出目录（包含 model 和 hyperparameters.json）
        model_type: 模型类型 ('baseline' 或 'adapter')
    
    Returns:
        PunctuationRestorer 实例
    """
    model_path = os.path.join(output_dir, 'model')
    hyperparameters_path = os.path.join(output_dir, 'hyperparameters.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(hyperparameters_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {hyperparameters_path}")
    
    return PunctuationRestorer(model_path, hyperparameters_path, model_type)


if __name__ == '__main__':
    # 测试示例
    import argparse
    
    parser = argparse.ArgumentParser(description='测试标点符号恢复器')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--hyperparameters', type=str, required=True,
                        help='超参数配置文件路径')
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'adapter'],
                        help='模型类型')
    parser.add_argument('--text', type=str, default='你好世界今天天气很好',
                        help='测试文本')
    
    args = parser.parse_args()
    
    # 创建恢复器
    restorer = PunctuationRestorer(
        args.model_path,
        args.hyperparameters,
        args.model_type
    )
    
    # 测试
    print(f"\n输入文本: {args.text}")
    result = restorer.restore_punctuation(args.text)
    print(f"输出文本: {result}")

