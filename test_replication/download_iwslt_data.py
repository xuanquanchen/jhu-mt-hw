# -*- coding: utf-8 -*-
"""
Download and format IWSLT Chinese dataset for punctuation restoration training

This script:
1. Downloads IWSLT TED Talks Chinese data from Hugging Face
2. Processes Chinese text into word + punctuation format
3. Handles expanded punctuation marks (，、。？！；《》)
4. Saves formatted data ready for training

Usage:
    python download_iwslt_data.py
"""

import os
import re
from collections import Counter
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: Please install datasets library:")
    print("  pip install datasets")
    exit(1)

# Punctuation marks to extract
CHINESE_PUNCTUATION = {
    '，': 'COMMA',
    '、': 'ENUMERATION',
    '。': 'PERIOD',
    '？': 'QUESTION',
    '！': 'EXCLAMATION',
    '；': 'SEMICOLON',
    '《': 'LEFT_BOOK_TITLE',
    '》': 'RIGHT_BOOK_TITLE',
}

# All punctuation marks as a set for quick lookup
PUNCT_SET = set(CHINESE_PUNCTUATION.keys())


def extract_punctuation_from_text(text):
    """
    Extract punctuation marks from Chinese text and return word-punctuation pairs.
    
    Args:
        text: Chinese text string
    
    Returns:
        List of (word, punctuation) tuples
    """
    results = []
    
    # Split text by whitespace first (if already tokenized)
    # Otherwise, process character by character
    words = text.split()
    
    for word in words:
        word = word.strip()
        if not word:
            continue
        
        # Check if word ends with punctuation
        punc = 'O'
        clean_word = word
        
        # Check for punctuation at the end
        if word and word[-1] in PUNCT_SET:
            punc = word[-1]
            clean_word = word[:-1]
        # Check for punctuation at the beginning (like 《)
        elif word and word[0] in PUNCT_SET:
            punc = word[0]
            clean_word = word[1:]
        
        # If we have a clean word, add it
        if clean_word:
            results.append((clean_word, punc))
        # If only punctuation (like standalone 《 or 》), add it as a word with punctuation
        elif punc != 'O':
            results.append((punc, 'O'))  # Punctuation mark itself as word, no punctuation after
    
    return results


def process_chinese_text_advanced(text):
    """
    Advanced processing: handle punctuation more intelligently.
    Handles cases like "《论语》" where punctuation is embedded.
    """
    results = []
    
    # Pattern to match Chinese characters and punctuation
    # This regex matches: (Chinese characters) + (optional punctuation)
    pattern = r'([\u4e00-\u9fff]+|[a-zA-Z0-9]+)([，、。？！；《》]?)'
    matches = re.findall(pattern, text)
    
    for word, punct in matches:
        word = word.strip()
        if word:
            # If punctuation is found, use it; otherwise 'O'
            punc = punct if punct else 'O'
            results.append((word, punc))
    
    # Also handle standalone punctuation marks
    # Find all punctuation marks that aren't attached to words
    for punc_mark in PUNCT_SET:
        # Look for standalone punctuation (with spaces around or at boundaries)
        pattern_standalone = rf'\s+{re.escape(punc_mark)}\s+'
        if re.search(pattern_standalone, text):
            # This is handled by the main pattern, but we check for edge cases
            pass
    
    return results


def format_data_for_training(word_punc_pairs, output_file):
    """
    Write word-punctuation pairs to file in training format.
    
    Args:
        word_punc_pairs: List of (word, punctuation) tuples
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, punc in word_punc_pairs:
            f.write(f"{word}\t{punc}\n")


def download_iwslt_chinese(output_dir='data/iwslt_chinese', years=['2014', '2015', '2016'], 
                          max_examples=None, use_advanced_processing=True):
    """
    Download and process IWSLT Chinese data.
    
    Args:
        output_dir: Directory to save processed data
        years: List of years to download ('2014', '2015', '2016')
        max_examples: Maximum number of examples per year (None for all)
        use_advanced_processing: Use advanced regex-based processing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_train = []
    all_valid = []
    all_test = []
    
    total_downloaded = 0
    
    print("=" * 60)
    print("Downloading IWSLT Chinese Dataset")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Years: {', '.join(years)}")
    print(f"Processing method: {'Advanced' if use_advanced_processing else 'Simple'}")
    print("=" * 60)
    
    for year in years:
        print(f"\n📥 Loading IWSLT {year}...")
        try:
            dataset = load_dataset("IWSLT/ted_talks_iwslt", 
                                 language_pair=("en", "zh-cn"), 
                                 year=year)
            
            # Process training data
            if 'train' in dataset:
                print(f"  Processing training data...")
                train_data = dataset['train']
                count = 0
                for example in tqdm(train_data, desc=f"  Year {year} train"):
                    if max_examples and count >= max_examples:
                        break
                    chinese_text = example['translation']['zh-cn']
                    if use_advanced_processing:
                        pairs = process_chinese_text_advanced(chinese_text)
                    else:
                        pairs = extract_punctuation_from_text(chinese_text)
                    all_train.extend(pairs)
                    count += 1
                print(f"  ✓ Processed {count} training examples")
            
            # Process validation data
            if 'validation' in dataset:
                print(f"  Processing validation data...")
                valid_data = dataset['validation']
                count = 0
                for example in tqdm(valid_data, desc=f"  Year {year} valid"):
                    chinese_text = example['translation']['zh-cn']
                    if use_advanced_processing:
                        pairs = process_chinese_text_advanced(chinese_text)
                    else:
                        pairs = extract_punctuation_from_text(chinese_text)
                    all_valid.extend(pairs)
                    count += 1
                print(f"  ✓ Processed {count} validation examples")
            
            # Process test data
            if 'test' in dataset:
                print(f"  Processing test data...")
                test_data = dataset['test']
                count = 0
                for example in tqdm(test_data, desc=f"  Year {year} test"):
                    chinese_text = example['translation']['zh-cn']
                    if use_advanced_processing:
                        pairs = process_chinese_text_advanced(chinese_text)
                    else:
                        pairs = extract_punctuation_from_text(chinese_text)
                    all_test.extend(pairs)
                    count += 1
                print(f"  ✓ Processed {count} test examples")
            
            total_downloaded += 1
            
        except Exception as e:
            print(f"  ✗ Error loading {year}: {e}")
            continue
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Data Statistics:")
    print("=" * 60)
    print(f"Training lines: {len(all_train):,}")
    print(f"Validation lines: {len(all_valid):,}")
    print(f"Test lines: {len(all_test):,}")
    
    # Count punctuation distribution
    punc_counts = Counter()
    for _, punc in all_train:
        punc_counts[punc] += 1
    
    print("\nPunctuation Distribution in Training Data:")
    for punc, count in punc_counts.most_common():
        punc_name = CHINESE_PUNCTUATION.get(punc, 'NO_PUNCTUATION')
        percentage = (count / len(all_train) * 100) if all_train else 0
        print(f"  {punc:3s} ({punc_name:20s}): {count:8,} ({percentage:5.2f}%)")
    
    # Save to files
    print("\n" + "=" * 60)
    print("Saving processed data...")
    print("=" * 60)
    
    if all_train:
        train_file = os.path.join(output_dir, 'train')
        format_data_for_training(all_train, train_file)
        print(f"✓ Saved {len(all_train):,} lines to {train_file}")
    
    if all_valid:
        valid_file = os.path.join(output_dir, 'valid')
        format_data_for_training(all_valid, valid_file)
        print(f"✓ Saved {len(all_valid):,} lines to {valid_file}")
    
    if all_test:
        test_file = os.path.join(output_dir, 'test')
        format_data_for_training(all_test, test_file)
        print(f"✓ Saved {len(all_test):,} lines to {test_file}")
    
    print("\n" + "=" * 60)
    print("✓ Download and processing complete!")
    print("=" * 60)
    print(f"\nData saved to: {output_dir}/")
    print("\nTo use this data:")
    print(f"  1. Copy files from {output_dir}/ to data/")
    print(f"  2. Or update your training script to use {output_dir}/")
    print("\nExample:")
    print(f"  cp {output_dir}/train data/train_iwslt")
    print(f"  cp {output_dir}/valid data/valid_iwslt")


def merge_with_existing_data(iwslt_dir='data/iwslt_chinese', existing_dir='data', 
                             output_dir='data/merged'):
    """
    Merge IWSLT data with existing training data.
    
    Args:
        iwslt_dir: Directory with IWSLT data
        existing_dir: Directory with existing data
        output_dir: Output directory for merged data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Merging IWSLT data with existing data")
    print("=" * 60)
    
    for split in ['train', 'valid', 'test']:
        iwslt_file = os.path.join(iwslt_dir, split)
        existing_file = os.path.join(existing_dir, split)
        output_file = os.path.join(output_dir, split)
        
        lines = []
        
        # Read existing data
        if os.path.exists(existing_file):
            with open(existing_file, 'r', encoding='utf-8') as f:
                lines.extend(f.readlines())
            print(f"  Loaded {len(lines):,} lines from existing {split}")
        
        # Read IWSLT data
        if os.path.exists(iwslt_file):
            with open(iwslt_file, 'r', encoding='utf-8') as f:
                iwslt_lines = f.readlines()
            lines.extend(iwslt_lines)
            print(f"  Added {len(iwslt_lines):,} lines from IWSLT {split}")
        
        # Write merged data
        if lines:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"  ✓ Saved {len(lines):,} total lines to {output_file}")
    
    print("\n✓ Merge complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and format IWSLT Chinese data')
    parser.add_argument('--output-dir', type=str, default='data/iwslt_chinese',
                       help='Output directory for processed data')
    parser.add_argument('--years', nargs='+', default=['2014', '2015', '2016'],
                       help='Years to download')
    parser.add_argument('--max-examples', type=int, default=None,
                       help='Maximum examples per year (None for all)')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple processing instead of advanced')
    parser.add_argument('--merge', action='store_true',
                       help='Merge with existing data in data/ directory')
    
    args = parser.parse_args()
    
    # Download data
    download_iwslt_chinese(
        output_dir=args.output_dir,
        years=args.years,
        max_examples=args.max_examples,
        use_advanced_processing=not args.simple
    )
    
    # Merge if requested
    if args.merge:
        print("\n")
        merge_with_existing_data(
            iwslt_dir=args.output_dir,
            existing_dir='data',
            output_dir='data/merged'
        )

