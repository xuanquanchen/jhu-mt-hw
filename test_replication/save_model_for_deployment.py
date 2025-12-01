# -*- coding: utf-8 -*-
"""
保存模型用于部署
将训练好的模型保存为适合部署的格式
"""

import os
import json
import shutil
import argparse
from pathlib import Path


def save_model_for_deployment(output_dir, deployment_dir=None, model_type='auto'):
    """
    保存模型用于部署
    
    Args:
        output_dir: 训练输出目录（包含 model 和 hyperparameters.json）
        deployment_dir: 部署目录（如果不提供，则使用 output_dir + '_deployment'）
        model_type: 模型类型 ('baseline', 'adapter', 或 'auto' 自动检测)
    """
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")
    
    model_path = os.path.join(output_dir, 'model')
    hyperparameters_path = os.path.join(output_dir, 'hyperparameters.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(hyperparameters_path):
        raise FileNotFoundError(f"超参数文件不存在: {hyperparameters_path}")
    
    # 读取超参数以确定模型类型
    with open(hyperparameters_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)
    
    if model_type == 'auto':
        model_type = hyperparams.get('model_type', 'BertChineseEmbSlimCNNlstmBert')
        if 'Adapter' in model_type:
            model_type = 'adapter'
        else:
            model_type = 'baseline'
    
    # 创建部署目录
    if deployment_dir is None:
        deployment_dir = output_dir + '_deployment'
    
    os.makedirs(deployment_dir, exist_ok=True)
    
    print("=" * 60)
    print("保存模型用于部署")
    print("=" * 60)
    print(f"源目录: {output_dir}")
    print(f"部署目录: {deployment_dir}")
    print(f"模型类型: {model_type}")
    print("=" * 60)
    
    # 复制模型文件
    print("\n复制模型文件...")
    shutil.copy2(model_path, os.path.join(deployment_dir, 'model.pth'))
    print("✓ 模型权重已复制")
    
    # 复制超参数文件
    print("\n复制超参数文件...")
    shutil.copy2(hyperparameters_path, os.path.join(deployment_dir, 'hyperparameters.json'))
    print("✓ 超参数文件已复制")
    
    # 创建部署配置文件
    print("\n创建部署配置...")
    deployment_config = {
        'model_type': model_type,
        'model_path': 'model.pth',
        'hyperparameters_path': 'hyperparameters.json',
        'version': '1.0.0',
        'description': 'Chinese Punctuation Restoration Model for Whisper Plugin'
    }
    
    config_path = os.path.join(deployment_dir, 'deployment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(deployment_config, f, indent=2, ensure_ascii=False)
    print("✓ 部署配置已创建")
    
    # 创建 README
    print("\n创建 README...")
    readme_content = f"""# 标点符号恢复模型 - 部署包

## 模型信息
- 模型类型: {model_type}
- 训练目录: {output_dir}
- 版本: {deployment_config['version']}

## 文件说明
- `model.pth`: 模型权重文件
- `hyperparameters.json`: 超参数配置
- `deployment_config.json`: 部署配置

## 使用方法

### Python 代码中使用

```python
from punctuation_restorer import create_restorer_from_output_dir

# 加载模型
restorer = create_restorer_from_output_dir(
    '{deployment_dir}',
    model_type='{model_type}'
)

# 处理文本
text = "你好世界今天天气很好"
result = restorer.restore_punctuation(text)
print(result)  # 输出: 你好世界，今天天气很好。
```

### 作为 Whisper 插件使用

```python
from whisper_plugin import WhisperPunctuationPlugin

# 初始化插件
plugin = WhisperPunctuationPlugin(
    output_dir='{deployment_dir}',
    model_type='{model_type}'
)

# 处理 Whisper 输出
whisper_text = "你好世界今天天气很好"
text_with_punctuation = plugin.process(whisper_text)
```

## 依赖
- torch
- transformers
- numpy

## 注意事项
- 首次使用时会自动下载 BERT 模型（bert-base-chinese）
- 建议使用 GPU 以获得更好的性能
- 模型文件较大，请确保有足够的存储空间
"""
    
    readme_path = os.path.join(deployment_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✓ README 已创建")
    
    # 复制必要的代码文件（可选）
    print("\n复制代码文件...")
    code_files = ['punctuation_restorer.py', 'whisper_plugin.py']
    for code_file in code_files:
        if os.path.exists(code_file):
            shutil.copy2(code_file, os.path.join(deployment_dir, code_file))
            print(f"✓ {code_file} 已复制")
    
    # 复制模型定义文件
    if model_type == 'adapter':
        if os.path.exists('model_adapter.py'):
            shutil.copy2('model_adapter.py', os.path.join(deployment_dir, 'model_adapter.py'))
            print("✓ model_adapter.py 已复制")
    
    if os.path.exists('model.py'):
        shutil.copy2('model.py', os.path.join(deployment_dir, 'model.py'))
        print("✓ model.py 已复制")
    
    print("\n" + "=" * 60)
    print("部署包创建完成！")
    print("=" * 60)
    print(f"\n部署目录: {deployment_dir}")
    print("\n文件列表:")
    for file in sorted(os.listdir(deployment_dir)):
        file_path = os.path.join(deployment_dir, file)
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  {file} ({size:.2f} MB)")
    
    print("\n使用说明:")
    print(f"  python whisper_plugin.py --output_dir {deployment_dir} --model_type {model_type} --text '测试文本'")


def main():
    parser = argparse.ArgumentParser(description='保存模型用于部署')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='训练输出目录（包含 model 和 hyperparameters.json）')
    parser.add_argument('--deployment_dir', type=str, default=None,
                        help='部署目录（如果不提供，则使用 output_dir + "_deployment"）')
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['auto', 'baseline', 'adapter'],
                        help='模型类型（auto 表示自动检测）')
    
    args = parser.parse_args()
    
    save_model_for_deployment(
        args.output_dir,
        args.deployment_dir,
        args.model_type
    )


if __name__ == '__main__':
    main()

