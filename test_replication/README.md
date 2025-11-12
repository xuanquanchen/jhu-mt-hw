# Chinese Punctuation Restoration - Clean Replication

This is a clean, minimal replication of the **BertChineseEmbSlimCNNlstmBert** model (the best model from the original repository).

## Structure

```
test_replication/
├── model.py          # Best model (BertChineseEmbSlimCNNlstmBert)
├── data_utils.py     # Data loading and preprocessing utilities
├── train.py          # Training script
├── test.py           # Test/Evaluation script
├── data/             # Data directory
│   ├── train         # Training data
│   ├── valid         # Validation data
│   └── test          # Test data
├── models/           # Saved models (created during training)
├── outputs/          # Training outputs (created during training)
└── README.md         # This file
```

## Data Format

Each line in the data files should be:
```
word punctuation
```

Example:
```
你好 O
世界 ，
今天 O
天气 O
很好 。
```

Supported punctuation marks:
- `O` - No punctuation
- `，` - Comma
- `。` - Period
- `？` - Question mark
- `！` - Exclamation mark
- `；` - Semicolon
- `、` - Enumeration comma

## Installation

```bash
pip install torch transformers numpy scikit-learn tqdm pandas
```

## Usage

### 1. Prepare Data

Place your data files in the `data/` directory:
- `data/train` - Training data
- `data/valid` - Validation data
- `data/test` - Test data

### 2. Train Model

```bash
python train.py
```

The script will:
- Load and preprocess data
- Initialize the model
- Train for 15 epochs
- Save the best model to `outputs/model_YYYYMMDD_HHMMSS/model`
- Save training progress to `outputs/model_YYYYMMDD_HHMMSS/progress.csv`

### 3. Test Model

```bash
python test.py \
    --model_path outputs/model_YYYYMMDD_HHMMSS/model \
    --hyperparameters outputs/model_YYYYMMDD_HHMMSS/hyperparameters.json \
    --test_data data/test \
    --output results.csv
```

## Model Architecture

**BertChineseEmbSlimCNNlstmBert** combines:
1. **BERT Embeddings** → CNN layers (word-level features)
2. **Second BERT** → Character-level features
3. **LSTM** → Sequence modeling
4. **Linear Layer** → Final classification

## Hyperparameters

Default settings (can be modified in `train.py`):
- Sequence length: 200
- Dropout: 0.1
- Epochs: 15
- Batch size: 10
- Learning rate: 1e-5
- Optimizer: AdamW

## Output

Training creates:
- `model` - Best model weights
- `hyperparameters.json` - Training configuration
- `progress.csv` - Training metrics (loss, accuracy, F1-scores)

Testing outputs:
- Per-class precision, recall, F1-scores
- Overall accuracy
- Macro-averaged metrics

## Notes

- Models are automatically downloaded from HuggingFace on first use
- Training on CPU is supported (but slower than GPU)
- The model uses `bert-base-chinese` for BERT layers
- Tokenizer uses `hfl/chinese-roberta-wwm-ext`

