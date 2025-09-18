import optparse
import sys
import math
from collections import defaultdict
import os

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards",
                     help="Data filename prefix (default=data/hansards)")
optparser.add_option("-e", "--english", dest="english", default="e",
                     help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f",
                     help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1000, type="int",
                     help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iterations", dest="iterations", default=8, type="int",
                     help="Number of EM iterations (default=8)")
optparser.add_option("-s", "--sigma", dest="sigma", default=0.3, type="float",
                     help="Diagonal bias parameter")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.01, type="float",
                     help="Alignment threshold")

optparser.add_option("--variants", dest="variants",
                     default="full,minus_len,minus_pos,diag_only,ibm1",
                     help="Comma-separated: full,minus_len,minus_pos,diag_only,ibm1")
optparser.add_option("--outdir", dest="outdir", default=".",
                     help="Directory to save .a files (default=.)")
optparser.add_option("--prefix", dest="prefix", default="ablation_",
                     help="Filename prefix (default=ablation_)")

(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Loading parallel corpus...\n")
bitext = [[sent.strip().split() for sent in pair]
          for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

def diag_bias(i, j_eff, f_len, e_len, sigma):
    if f_len <= 1 or e_len <= 1:
        return 1.0
    ni = i / (f_len - 1.0)
    nj = j_eff / (e_len - 1.0)
    distance = abs(ni - nj)
    return math.exp(-(distance ** 2) / (2.0 * sigma ** 2))

def pos_bias(i, j_eff, f_len, e_len):
    if f_len <= 1 or e_len <= 1:
        return 1.0
    ni = i / (f_len - 1.0)
    nj = j_eff / (e_len - 1.0)
    return math.exp(-2.0 * abs(ni - nj))

def len_bias(f_len, e_len):
    ratio = f_len / max(e_len, 1)
    return 1.0 if 0.5 <= ratio <= 2.0 else 0.5

def combined_bias(i, j_eff, f_len, e_len, sigma, use_diag, use_pos, use_len):
    b = 1.0
    if use_diag: b *= diag_bias(i, j_eff, f_len, e_len, sigma)
    if use_pos:  b *= pos_bias(i, j_eff, f_len, e_len)
    if use_len:  b *= len_bias(f_len, e_len)
    return b

VARIANTS = {
    "full":       (True,  True,  True ),
    "minus_len":  (True,  True,  False),
    "minus_pos":  (True,  False, True ),
    "diag_only":  (True,  False, False),
    "ibm1":       (False, False, False),
}

def train_and_decode(use_diag, use_pos, use_len):
    f_vocab, e_vocab = set(), set()
    for (f, e) in bitext:
        f_vocab.update(f); e_vocab.update(e)
    t = defaultdict(lambda: defaultdict(float))
    invE = 1.0 / float(len(e_vocab))
    for f in f_vocab:
        for e in e_vocab:
            t[f][e] = invE

    # EM
    for _ in range(opts.iterations):
        count_fe = defaultdict(lambda: defaultdict(float))
        count_e  = defaultdict(float)

        for (f, e) in bitext:
            e_with_null = ['NULL'] + e
            f_len, e_len = len(f), len(e)

            for i, f_word in enumerate(f):
                total = 0.0
                for j, e_word in enumerate(e_with_null):
                    if j == 0:
                        w = t[f_word][e_word]
                    else:
                        cb = combined_bias(i, j-1, f_len, e_len, opts.sigma, use_diag, use_pos, use_len)
                        w = t[f_word][e_word] * cb
                    total += w

                if total <= 0.0:
                    continue

                for j, e_word in enumerate(e_with_null):
                    if j == 0:
                        expected = t[f_word][e_word] / total
                    else:
                        cb = combined_bias(i, j-1, f_len, e_len, opts.sigma, use_diag, use_pos, use_len)
                        expected = (t[f_word][e_word] * cb) / total
                    count_fe[f_word][e_word] += expected
                    count_e[e_word] += expected

        # M-step
        new_t = defaultdict(lambda: defaultdict(float))
        for f_word in count_fe:
            for e_word in count_fe[f_word]:
                denom = count_e[e_word]
                if denom > 0.0:
                    new_t[f_word][e_word] = count_fe[f_word][e_word] / denom
        t = new_t

    # Decode
    lines = []
    for (f, e) in bitext:
        f_len, e_len = len(f), len(e)
        out = []
        for i, f_word in enumerate(f):
            best_alignment = 0
            best_score = t[f_word]['NULL']
            for j, e_word in enumerate(e):
                cb = combined_bias(i, j, f_len, e_len, opts.sigma, use_diag, use_pos, use_len)
                score = t[f_word][e_word] * cb
                if score > best_score:
                    best_score = score
                    best_alignment = j
            if best_score > opts.threshold:
                out.append(f"{i}-{best_alignment}")
        lines.append(" ".join(out))
    return "\n".join(lines)

names = [v.strip() for v in opts.variants.split(",") if v.strip() in VARIANTS]
if not names:
    sys.stderr.write("[ERR] No valid variants specified.\n")
    sys.exit(1)

os.makedirs(opts.outdir, exist_ok=True)
sys.stderr.write(f"Running variants: {', '.join(names)}\n")
for name in names:
    use_diag, use_pos, use_len = VARIANTS[name]
    sys.stderr.write(f"[Run] {name}  n={opts.num_sents}  iters={opts.iterations}  sigma={opts.sigma}  thr={opts.threshold}\n")
    alignment = train_and_decode(use_diag, use_pos, use_len)
    outfile = os.path.join(opts.outdir, f"{opts.prefix}{name}.a")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(alignment + "\n")
    sys.stderr.write(f"  -> wrote {outfile}\n")
