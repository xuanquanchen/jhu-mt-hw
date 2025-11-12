# Download IWSLT Chinese Dataset

This guide explains how to download and format additional Chinese data from IWSLT for training.

## Quick Start

```bash
# Install required library
pip install datasets

# Download all IWSLT Chinese data (2014, 2015, 2016)
python download_iwslt_data.py

# Download with limited examples (faster, for testing)
python download_iwslt_data.py --max-examples 100

# Download only specific years
python download_iwslt_data.py --years 2014 2015

# Merge with existing data
python download_iwslt_data.py --merge
```

## What It Does

1. **Downloads** IWSLT TED Talks Chinese data from Hugging Face
2. **Processes** Chinese text into `word + punctuation` format
3. **Handles** all punctuation marks: `，、。？！；《》`
4. **Saves** formatted data ready for training

## Output Format

The script creates files in `data/iwslt_chinese/`:
- `train` - Training data
- `valid` - Validation data
- `test` - Test data

Each line format: `word\tpunctuation`

Example:
```
你好	O
世界	，
今天	O
天气	O
很好	。
```

## Usage Options

### Basic Usage
```bash
python download_iwslt_data.py
```
Downloads all years (2014, 2015, 2016) and saves to `data/iwslt_chinese/`

### Limited Examples (for testing)
```bash
python download_iwslt_data.py --max-examples 50
```
Downloads only 50 examples per year (faster for testing)

### Specific Years
```bash
python download_iwslt_data.py --years 2014
```
Downloads only 2014 data

### Custom Output Directory
```bash
python download_iwslt_data.py --output-dir data/my_iwslt
```

### Merge with Existing Data
```bash
python download_iwslt_data.py --merge
```
Merges IWSLT data with existing `data/train`, `data/valid`, `data/test` files

### Simple Processing (faster, less accurate)
```bash
python download_iwslt_data.py --simple
```
Uses simpler text processing (may miss some punctuation)

## After Downloading

### Option 1: Use IWSLT data directly
```bash
# Copy to main data directory
cp data/iwslt_chinese/train data/train_iwslt
cp data/iwslt_chinese/valid data/valid_iwslt
cp data/iwslt_chinese/test data/test_iwslt

# Update your training script to use these files
```

### Option 2: Merge with existing data
```bash
# Use --merge flag when downloading
python download_iwslt_data.py --merge

# Merged data will be in data/merged/
# Copy to main data directory
cp data/merged/* data/
```

### Option 3: Replace existing data
```bash
# Backup existing data first
mv data/train data/train_backup
mv data/valid data/valid_backup
mv data/test data/test_backup

# Copy IWSLT data
cp data/iwslt_chinese/* data/
```

## Data Statistics

After downloading, the script will show:
- Total lines per split (train/valid/test)
- Punctuation distribution
- File locations

Example output:
```
Data Statistics:
============================================================
Training lines: 125,432
Validation lines: 8,234
Test lines: 7,891

Punctuation Distribution in Training Data:
  O  (NO_PUNCTUATION    ):   98,234 (78.35%)
  ， (COMMA             ):   15,234 (12.14%)
  。 (PERIOD            ):    8,234 ( 6.56%)
  ？ (QUESTION          ):    1,234 ( 0.98%)
  ！ (EXCLAMATION       ):      234 ( 0.19%)
  ； (SEMICOLON         ):      234 ( 0.19%)
  、 (ENUMERATION       ):      234 ( 0.19%)
  《 (LEFT_BOOK_TITLE   ):      123 ( 0.10%)
  》 (RIGHT_BOOK_TITLE  ):      123 ( 0.10%)
```

## Troubleshooting

### Error: "No module named 'datasets'"
```bash
pip install datasets
```

### Error: "Connection timeout"
- Check your internet connection
- Try downloading one year at a time: `--years 2014`

### Data format issues
- Make sure the output files use tab-separated format (`word\tpunctuation`)
- Check that punctuation marks are correctly extracted

### Out of memory
- Use `--max-examples` to limit data size
- Download one year at a time

## Notes

- **Processing time**: Full dataset (all years) may take 10-30 minutes
- **Data size**: Full dataset can be 100MB+ of text
- **Punctuation**: The script handles all standard Chinese punctuation including 《 and 》
- **Format**: Output matches your training script's expected format

## Integration with Training

After downloading, you can use the data with any training script:

```bash
# Train with IWSLT data
python train_v3.py  # Will use data/train, data/valid

# Or specify custom data location
# (modify train_v3.py to point to data/iwslt_chinese/)
```

