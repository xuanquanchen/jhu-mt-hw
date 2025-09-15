#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iterations", dest="iterations", default=5, type="int", help="Number of EM iterations (default=5)")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training IBM Model 1...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

# Initialize vocabulary and translation probabilities
f_vocab = set()
e_vocab = set()
for (f, e) in bitext:
  f_vocab.update(f)
  e_vocab.update(e)

t = defaultdict(lambda: defaultdict(float))
for f in f_vocab:
  for e in e_vocab:
    t[f][e] = 1.0 / len(e_vocab)

# EM iterations
for iteration in range(opts.iterations):
  sys.stderr.write("../../../../Desktop/MT backup/jhu-mt-hw/hw2")
  count_fe = defaultdict(lambda: defaultdict(float))
  count_e = defaultdict(float)
  
  for (f, e) in bitext:
    e_with_null = ['NULL'] + e
    
    for f_word in f:
      total = sum(t[f_word][e_word] for e_word in e_with_null)
      
      if total > 0:
        for e_word in e_with_null:
          expected_count = t[f_word][e_word] / total
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
  for (i, f_word) in enumerate(f):
    best_alignment = 0
    best_prob = t[f_word]['NULL']
    for (j, e_word) in enumerate(e):
      if t[f_word][e_word] > best_prob:
        best_prob = t[f_word][e_word]
        best_alignment = j
    if best_prob > 0.01:
      sys.stdout.write("%i-%i " % (i, best_alignment))
  sys.stdout.write("\n")
