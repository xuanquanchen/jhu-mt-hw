There are three python programs here (`-h` for usage):

- `./align` aligns words.

- `./check-alignments` checks that the entire dataset is aligned, and
  that there are no out-of-bounds alignment points.

- `./score-alignments` computes alignment error rate.

The commands work in a pipeline. For instance:

   > ./align -t 0.9 -n 1000 | ./check | ./grade -n 5

The `data` directory contains a fragment of the Canadian Hansards,
aligned by Ulrich Germann:

- `hansards.e` is the English side.

- `hansards.f` is the French side.

- `hansards.a` is the alignment of the first 37 sentences. The 
  notation i-j means the word as position i of the French is 
  aligned to the word at position j of the English. Notation 
  i?j means they are probably aligned. Positions are 0-indexed.

# Update

## File Orgnizations

- `ibm_model1.py` - IBM Model 1 implementation
- `ibm_model1.a` - Alignment file for IBM Model 1 (1000 lines)
- `diagonal_model.py` - Diagonal preference model implementation
- `diagonal_model.a` - Diagonal alignment output (1000 lines)
- `hybrid_model.py` - Hybrid model(best performance)
- `alignment` - Final alignment output 10000 lines (best model)
- `README.md` - This file
- `hybrid_hyperparameters.py` - itenerate through sigma and threshold combinations to find the best combination
- `hybrid_hyperparameters_output.txt` - output of the precision, recall, AER from `hybrid_hyperparameters.py`.

## Models Implemented

### 1. IBM Model 1 (`ibm_model1.py`)
- **Algorithm**: IBM Model 1 with EM training
- **Performance**: AER = 0.436 (significant improvement)
- **Description**: Probabilistic model using expectation-maximization
- **Format**: Linear script matching `align` structure

### 2. Diagonal Preference Model (`diagonal_model.py`)
- **Algorithm**: IBM Model 1 + diagonal bias
- **Performance**: AER = 0.383
- **Description**: Combines IBM Model 1 with Gaussian diagonal bias
- **Format**: Linear script matching `align` structure

### 3. Hybrid Model (`hybrid_model.py`) **BEST PERFORMING**
- **Algorithm**: IBM Model 1 + multiple bias terms
- **Performance**: AER = 0.261 (62% improvement over baseline)
- **Description**: Combines translation probabilities with diagonal, position, and length biases
- **Format**: Linear script matching `align` structure

## Performance Summary

| Model | Precision | Recall    | AER       | Improvement    |
|-------|-----------|-----------|-----------|----------------|
| Baseline Dice | 0.239     | 0.595     | 0.682     | -              |
| IBM Model 1 | 0.513     | 0.675     | 0.436     | 36% better     |
| Diagonal Preference | 0.562     | 0.734     | 0.383     | 44% better     |
| **Hybrid Model** | **0.648** | **0.766** | **0.314** | **54% better** |

## Usage

All models follow the same command-line interface and output format for consistency.

### Running Individual Models

```bash
# IBM Model 1
python ibm1.py -n 10000 -i 8 > ibm1.a

# Diagonal Preference Model
python diagonal.py -n 10000 -i 8 -s 1.0 > diagonal.a

# Hybrid Model (recommended - best performance)
python hybrid.py -n 10000 -i 8 -s 0.3 -t 0.01 > alignment
```

### Evaluating Alignments

```bash
python score-alignments < alignment_file.a
```

```bash
# To print out precision, recell, AER results (1000 lines alignment example)
python score-alignments < alignment_file | tail -3
```
### Checking Alignment Format

```bash
python check-alignments < alignment_file
```

## Final Submission

The final alignment file `alignment` was generated using the **Hybrid Model** with the following parameters:
- Training sentences: All available data
- EM iterations: 8
- Diagonal bias parameter (σ): 0.3
- threshold: 0.01
- Multiple bias terms: diagonal, position, and length biases