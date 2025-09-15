#!/usr/bin/env python
import optparse
import sys
import math
from collections import defaultdict
import subprocess

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1000, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iterations", dest="iterations", default=8, type="int", help="Number of EM iterations (default=8)")
(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# Load bitext
sys.stderr.write("Loading parallel corpus...\n")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

# Define ranges for sigma and threshold
sigma_values = [0.4, 0.3, 0.2, 0.1]
threshold_values = [round(x * 0.001, 3) for x in range(1, 11)]

def run_alignment(sigma, threshold):
    # Initialize vocabulary and translation probabilities
    f_vocab = set()
    e_vocab = set()
    for (f, e) in bitext:
        f_vocab.update(f)
        e_vocab.update(e)
    t = defaultdict(lambda: defaultdict(float))
    for f_word in f_vocab:
        for e_word in e_vocab:
            t[f_word][e_word] = 1.0 / len(e_vocab)

    # EM iterations
    for iteration in range(opts.iterations):
        count_fe = defaultdict(lambda: defaultdict(float))
        count_e = defaultdict(float)
        for (f, e) in bitext:
            e_with_null = ['NULL'] + e
            f_len = len(f)
            e_len = len(e)
            for i, f_word in enumerate(f):
                total = 0.0
                for j, e_word in enumerate(e_with_null):
                    if j == 0:
                        total += t[f_word][e_word]
                    else:
                        norm_i = i / max(f_len - 1, 1)
                        norm_j = (j - 1) / max(e_len - 1, 1)
                        diag_bias = math.exp(-((norm_i - norm_j) ** 2) / (2 * sigma ** 2))
                        pos_bias = math.exp(-abs(norm_i - norm_j) * 2)
                        len_bias = 1.0 if 0.5 <= f_len / max(e_len, 1) <= 2.0 else 0.5
                        total += t[f_word][e_word] * diag_bias * pos_bias * len_bias
                if total > 0:
                    for j, e_word in enumerate(e_with_null):
                        if j == 0:
                            expected_count = t[f_word][e_word] / total
                        else:
                            norm_i = i / max(f_len - 1, 1)
                            norm_j = (j - 1) / max(e_len - 1, 1)
                            diag_bias = math.exp(-((norm_i - norm_j) ** 2) / (2 * sigma ** 2))
                            pos_bias = math.exp(-abs(norm_i - norm_j) * 2)
                            len_bias = 1.0 if 0.5 <= f_len / max(e_len, 1) <= 2.0 else 0.5
                            expected_count = (t[f_word][e_word] * diag_bias * pos_bias * len_bias) / total
                        count_fe[f_word][e_word] += expected_count
                        count_e[e_word] += expected_count
        # Update translation probabilities
        new_t = defaultdict(lambda: defaultdict(float))
        for f_word in count_fe:
            for e_word in count_fe[f_word]:
                if count_e[e_word] > 0:
                    new_t[f_word][e_word] = count_fe[f_word][e_word] / count_e[e_word]
        t = new_t

    # Generate alignment as string
    alignment_lines = []
    for (f, e) in bitext:
        f_len = len(f)
        e_len = len(e)
        line = []
        for i, f_word in enumerate(f):
            best_alignment = 0
            best_score = t[f_word]['NULL']
            for j, e_word in enumerate(e):
                trans_prob = t[f_word][e_word]
                norm_i = i / max(f_len - 1, 1)
                norm_j = j / max(e_len - 1, 1)
                diag_bias = math.exp(-((norm_i - norm_j) ** 2) / (2 * sigma ** 2))
                pos_bias = math.exp(-abs(norm_i - norm_j) * 2)
                len_bias = 1.0 if 0.5 <= f_len / max(e_len, 1) <= 2.0 else 0.5
                score = trans_prob * diag_bias * pos_bias * len_bias
                if score > best_score:
                    best_score = score
                    best_alignment = j
            if best_score > threshold:
                line.append(f"{i}-{best_alignment}")
        alignment_lines.append(" ".join(line))
    return "\n".join(alignment_lines)

# Main loop: run alignment, pipe to score-alignments, capture last 3 lines
with open("hybrid_hyperparameters_output.txt", "w") as out_file:
    for sigma in sigma_values:
        for threshold in threshold_values:
            sys.stderr.write(f"Processing sigma={sigma}, threshold={threshold}\n")
            alignment_str = run_alignment(sigma, threshold)
            # Run score-alignments via subprocess
            score_proc = subprocess.Popen(
                ["python", "score-alignments"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            output, _ = score_proc.communicate(input=alignment_str.encode("utf-8"))
            output_lines = output.decode("utf-8").splitlines()[-3:]  # last 3 lines
            out_file.write(f"# sigma={sigma}, threshold={threshold}\n")
            out_file.write("\n".join(output_lines) + "\n\n")
