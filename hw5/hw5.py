#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division

import argparse
import imp
import logging
import random
from torch.optim.lr_scheduler import StepLR
import time
from io import open

import matplotlib
#if you are running on the gradx/ugradx/ another cluster, 
#you will need the following line
#if you run on a local machine, you can comment it out
#matplotlib.use('agg')
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import corpus_bleu
from torch import dropout, optim


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to confict with other people's jobs.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device:', device)
#device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15
attention_plot_counter = 1


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    indexes = []
    for word in sentence.split():
        if word in vocab.word2index:
            indexes.append(vocab.word2index[word])
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device)




def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################


class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        [DONE] You should make your LSTM modular and re-use it in the Decoder.
        """
        # initialize word embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        # initialize bidirectional LSTM layer
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        if bidirectional:
            self.reduce = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.reduce = nn.Identity()

    def forward(self, input_batch, lengths=None):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        if lengths is None:
            if input_batch.dim() == 1:
                input_batch = input_batch.unsqueeze(0)  # [1, seq_len]
                lengths = torch.tensor([input_batch.size(1)], device=device)
            elif input_batch.dim() == 2 and input_batch.size(1) == 1:
                input_batch = input_batch.transpose(0, 1)
                lengths = torch.tensor([input_batch.size(1)], device=device)
        # get word embeddings
        embedded = self.embedding(input_batch)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        outputs = self.reduce(outputs)
        
        return outputs, (hidden, cell)

    def get_initial_hidden_state(self):
        h0 = torch.zeros(1, 1, self.hidden_size, device=device)
        c0 = torch.zeros(1, 1, self.hidden_size, device=device)
        return (h0, c0)


class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.3, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        "*** YOUR CODE HERE ***"
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, hidden, encoder_outputs):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """
        
        "*** YOUR CODE HERE ***"
        # fix translate bug when hidden is a tensor
        h,c = hidden
        emb = self.dropout(self.embedding(input_step))
        
        h_t = h[-1].unsqueeze(1) 
        
        # attention
        repeat_h = h_t.expand(-1, encoder_outputs.size(1), -1)
        energy = torch.tanh(self.attn(torch.cat((repeat_h, encoder_outputs), dim=2)))
        attn_scores = self.v(energy).squeeze(2)        # [batch,src_len]
        attn_weights = F.softmax(attn_scores, dim=1)   # [batch,src_len]
        
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((emb, context), dim=2)
        
        rnn_input = self.attn_combine(rnn_input)
        rnn_input = torch.tanh(rnn_input)
        
        output, hidden = self.lstm(rnn_input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden, attn_weights
    
    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
def batchify(pairs, src_vocab, tgt_vocab, batch_size):
    random.shuffle(pairs)
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        src_tensors = [tensor_from_sentence(src_vocab, p[0]) for p in batch_pairs]
        tgt_tensors = [tensor_from_sentence(tgt_vocab, p[1]) for p in batch_pairs]
        src_lens = torch.tensor([len(t) for t in src_tensors])
        tgt_lens = torch.tensor([len(t) for t in tgt_tensors])

        src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=EOS_index)
        tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=EOS_index)
        yield src_padded, src_lens, tgt_padded, tgt_lens


def train_epoch(encoder, decoder, pairs, src_vocab, tgt_vocab, optimizer, criterion, batch_size, tf_ratio):
    start_time = time.time()
    total_sentences = 0
    total_loss = 0

    for src_batch, src_lens, tgt_batch, tgt_lens in batchify(pairs, src_vocab, tgt_vocab, batch_size):
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(src_batch, src_lens)
        decoder_input = torch.full((src_batch.size(0), 1), SOS_index, device=device, dtype=torch.long)
        decoder_hidden = encoder_hidden

        max_tgt_len = tgt_batch.size(1)
        loss = 0
        for t in range(max_tgt_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, tgt_batch[:, t])
            teacher_force = random.random() < tf_ratio
            top1 = decoder_output.argmax(1).unsqueeze(1)
            decoder_input = tgt_batch[:, t].unsqueeze(1) if teacher_force else top1

        loss.backward()
        optimizer.step()
        total_loss += loss.item() / max_tgt_len
        total_sentences += src_batch.size(0)

    elapsed = time.time() - start_time
    print(f"Epoch finished: {total_sentences / elapsed:.2f} sentences/sec")
    return total_loss / len(pairs)


######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()
    beam_size = 10
    
    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence).unsqueeze(0)
        input_length = torch.tensor([input_tensor.size(1)], device=device)
        
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_length)
        encoder_outputs = encoder_outputs  # [1, seq_len, H]

        decoder_input = torch.full((1, 1), SOS_index, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        beams = [(0.0, [SOS_index], decoder_hidden, torch.zeros(max_length, input_length.item(), device=device))]

        completed = []
        
        for _ in range(max_length):
            new_beams = []
            for log_prob, seq, hidden, attn_history in beams:
                if seq[-1] == EOS_index:
                    completed.append((log_prob, seq, attn_history))
                    continue
            
                decoder_input = torch.tensor([[seq[-1]]], device=device)
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, hidden, encoder_outputs)
                
                log_probs = F.log_softmax(decoder_output, dim=1)
                topv, topi = log_probs.topk(beam_size)

                for i in range(beam_size):
                    next_token = topi[0][i].item()
                    prob = topv[0][i].item()
                    new_seq = seq + [next_token]
                    length_penalty = ((5 + len(seq)) / 6)
                    new_log_prob = (log_prob + prob) / length_penalty
                    new_attn = attn_history.clone()

                    if decoder_attention is not None:
                        attn_len = min(new_attn.size(1), decoder_attention.size(0))
                        new_attn[len(seq)-1, :attn_len] = decoder_attention[:attn_len]
                    
                    new_beams.append((new_log_prob, new_seq, decoder_hidden, new_attn))
                    
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

        if not completed:
            completed = beams

        best = max(completed, key=lambda x: x[0])
        best_seq = best[1]
        decoded_words = [tgt_vocab.index2word[idx] for idx in best_seq[1:] if idx not in [EOS_index, SOS_index]]
        attentions = best[2]
        
        return decoded_words, attentions


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    global attention_plot_counter
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    attn = attentions.cpu().numpy()
    inWords = input_sentence.split()
    outWords = [w for w in output_words if w not in {SOS_token, EOS_token}]
    
    n_target = len(outWords)
    n_source = len(inWords)
    
    attn = attn[:n_target, :n_source]

    cax = ax.matshow(attn, cmap='viridis')
    ax.set_xticks(range(n_source))
    ax.set_yticks(range(n_target))
    ax.set_xticklabels(inWords, rotation=90)
    ax.set_yticklabels(outWords)
    ax.set_xlabel('Source Words')
    ax.set_ylabel('Target Words')
    ax.set_title('Attention Weights')
    fig.colorbar(cax)

    fig_name = f'attention_plot_{attention_plot_counter}.png'
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Attention plot saved as {fig_name}')
    attention_plot_counter += 1


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    global attention_plot_counter
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=1024, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=150000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=2000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.0003, type=float,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='translations', # changed from out.txt to translations
                    help='output file for test translations_beamsearch')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()
    
    random.seed(42)
    torch.manual_seed(42)

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.AdamW(params, lr=args.initial_learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.85)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    tf_start = 0.95
    tf_end = 0.5
    tf_decay_iters = args.n_iters * 0.9

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    num_epochs = 1
    batch_size = 32

    for epoch in range(num_epochs):
        avg_loss = train_epoch(
            encoder, decoder, train_pairs,
            src_vocab, tgt_vocab,
            optimizer, criterion,
            batch_size, tf_ratio=0.9
        )
        print(f"[Epoch {epoch+1}] Avg loss: {avg_loss:.4f}")
        scheduler.step()

    state = {
        'epoch': epoch,
        'enc_state': encoder.state_dict(),
        'dec_state': decoder.state_dict(),
        'opt_state': optimizer.state_dict(),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
    }
    filename = f'state_epoch_{epoch+1:03d}.pt'
    torch.save(state, filename)
    logging.debug(f'wrote checkpoint to {filename}')

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
