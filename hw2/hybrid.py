#!/usr/bin/env python
import optparse
import sys
import math
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iterations", dest="iterations", default=8, type="int", help="Number of EM iterations (default=8)")
optparser.add_option("-s", "--sigma", dest="sigma", default=1.0, type="float", help="Diagonal bias parameter (default=1.0)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.01, type="float", help="Alignment threshold (default=0.01)")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training hybrid alignment model...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

# initialize vocabulary and translation probabilities
f_vocab = set()
e_vocab = set()
for (f, e) in bitext:
  f_vocab.update(f)
  e_vocab.update(e)

t = defaultdict(lambda: defaultdict(float))
for f in f_vocab:
  for e in e_vocab:
    t[f][e] = 1.0 / len(e_vocab)

# estimator: EM iterations
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
          # Diagonal bias
          norm_i = i / max(f_len - 1, 1)
          norm_j = (j-1) / max(e_len - 1, 1)
          distance = abs(norm_i - norm_j)
          diag_bias = math.exp(-(distance ** 2) / (2 * opts.sigma ** 2))
          
          # Position bias
          f_density = i / max(f_len - 1, 1)
          e_density = (j-1) / max(e_len - 1, 1)
          density_diff = abs(f_density - e_density)
          pos_bias = math.exp(-density_diff * 2)
          
          # Length bias
          len_ratio = f_len / max(e_len, 1)
          if 0.5 <= len_ratio <= 2.0:
            len_bias = 1.0
          else:
            len_bias = 0.5
          
          combined_bias = diag_bias * pos_bias * len_bias
          total += t[f_word][e_word] * combined_bias
      
      if total > 0:
        for j, e_word in enumerate(e_with_null):
          if j == 0:
            expected_count = t[f_word][e_word] / total
          else:
            # Diagonal bias
            norm_i = i / max(f_len - 1, 1)
            norm_j = (j-1) / max(e_len - 1, 1)
            distance = abs(norm_i - norm_j)
            diag_bias = math.exp(-(distance ** 2) / (2 * opts.sigma ** 2))
            
            # Position bias
            f_density = i / max(f_len - 1, 1)
            e_density = (j-1) / max(e_len - 1, 1)
            density_diff = abs(f_density - e_density)
            pos_bias = math.exp(-density_diff * 2)
            
            # Length bias
            len_ratio = f_len / max(e_len, 1)
            if 0.5 <= len_ratio <= 2.0:
              len_bias = 1.0
            else:
              len_bias = 0.5
            
            combined_bias = diag_bias * pos_bias * len_bias
            expected_count = (t[f_word][e_word] * combined_bias) / total
          
          count_fe[f_word][e_word] += expected_count
          count_e[e_word] += expected_count
  
  # Update translation probabilities
  new_t = defaultdict(lambda: defaultdict(float))
  for f_word in count_fe:
    for e_word in count_fe[f_word]:
      if count_e[e_word] > 0:
        new_t[f_word][e_word] = count_fe[f_word][e_word] / count_e[e_word]
  t = new_t

sys.stderr.write("\n")

for (f, e) in bitext:
  f_len = len(f)
  e_len = len(e)
  for (i, f_word) in enumerate(f):
    best_alignment = 0
    best_score = t[f_word]['NULL']
    for (j, e_word) in enumerate(e):
      # Translation probability
      trans_prob = t[f_word][e_word]
      
      # Diagonal bias
      norm_i = i / max(f_len - 1, 1)
      norm_j = j / max(e_len - 1, 1)
      distance = abs(norm_i - norm_j)
      diag_bias = math.exp(-(distance ** 2) / (2 * opts.sigma ** 2))
      
      # Position bias
      f_density = i / max(f_len - 1, 1)
      e_density = j / max(e_len - 1, 1)
      density_diff = abs(f_density - e_density)
      pos_bias = math.exp(-density_diff * 2)
      
      # Length bias
      len_ratio = f_len / max(e_len, 1)
      if 0.5 <= len_ratio <= 2.0:
        len_bias = 1.0
      else:
        len_bias = 0.5
      
      # Combined score
      score = trans_prob * diag_bias * pos_bias * len_bias
      
      if score > best_score:
        best_score = score
        best_alignment = j
    if best_score > opts.threshold:
      sys.stdout.write("%i-%i " % (i, best_alignment))
  sys.stdout.write("\n")
