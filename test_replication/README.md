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

**Standard Set (train.py, train_v1.py, train_v2.py):**
- `O` - No punctuation
- `，` - Comma
- `。` - Period
- `？` - Question mark
- `！` - Exclamation mark
- `；` - Semicolon
- `、` - Enumeration comma

**Expanded Set (train_v3.py):**
- All standard marks above, plus:
- `《` - Left book title mark
- `》` - Right book title mark

## Installation

```bash
# Core dependencies
pip install torch transformers numpy scikit-learn tqdm pandas

# For downloading IWSLT data (optional)
pip install datasets
```

## Usage

### 1. Prepare Data

#### Option A: Use Existing Data
Place your data files in the `data/` directory:
- `data/train` - Training data
- `data/valid` - Validation data
- `data/test` - Test data

#### Option B: Download IWSLT Data
Download additional Chinese data from IWSLT:

```bash
# Install datasets library
pip install datasets

# Download IWSLT Chinese data
python download_iwslt_data.py

# Or download with limited examples (for testing)
python download_iwslt_data.py --max-examples 100

# Merge with existing data
python download_iwslt_data.py --merge
```

See `DOWNLOAD_DATA.md` for detailed instructions.

### 2. Train Model

#### Available Training Scripts

**`train.py`** - Original full training script (15 epochs, full dataset)

**`train_v1.py`** - Quick test version
- 1 epoch only
- Limited to first 99,998 training lines
- No class weights (baseline)

**`train_v2.py`** - With class weights
- 1 epoch only
- Limited to first 99,998 training lines
- **Computes class weights from data** to handle imbalanced classes
- Better for learning punctuation marks

**`train_v3.py`** - Class weights + validation cap ⭐ **Recommended for testing**
- 1 epoch only
- Limited to first 99,998 training lines
- **Limited to first 20,000 validation lines** (faster validation)
- **Computes class weights from data** to handle imbalanced classes
- Faster overall training/validation cycle

```bash
# Use V3 for faster testing (with validation cap)
python train_v3.py

# Or use V2 for standard training with class weights
python train_v2.py

# Or use original for full training
python train.py
```

The scripts will:
- Load and preprocess data
- Initialize the model
- Train for specified epochs
- Save the best model to `outputs/model_vX_YYYYMMDD_HHMMSS/model`
- Save training progress to `outputs/model_vX_YYYYMMDD_HHMMSS/progress.csv`

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

