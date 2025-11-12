# Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
cd test_replication
pip install -r requirements.txt
```

2. **Prepare data:**
   - Place your data files in `data/` directory:
     - `data/train`
     - `data/valid`
     - `data/test`
   - Format: one line per word: `word punctuation`

## Training

```bash
python train.py
```

This will:
- Train the model for 15 epochs
- Save best model to `outputs/model_YYYYMMDD_HHMMSS/model`
- Save training progress to `outputs/model_YYYYMMDD_HHMMSS/progress.csv`

## Testing

After training, test the model:

```bash
python test.py \
    --model_path outputs/model_YYYYMMDD_HHMMSS/model \
    --hyperparameters outputs/model_YYYYMMDD_HHMMSS/hyperparameters.json \
    --test_data data/test
```

## Files

- `model.py` - Best model (BertChineseEmbSlimCNNlstmBert)
- `data_utils.py` - Data loading utilities
- `train.py` - Training script
- `test.py` - Evaluation script
- `data/` - Your data files go here
- `outputs/` - Training outputs (created automatically)

## Model

**BertChineseEmbSlimCNNlstmBert** - The best performing model:
- BERT embeddings + CNN (word-level features)
- Second BERT (character-level features)
- LSTM (sequence modeling)
- Linear classification layer

## Example Data Format

```
你好 O
世界 ，
今天 O
天气 O
很好 。
```

That's it! Simple and clean. 🚀

