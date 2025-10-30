#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
from torch.optim.lr_scheduler import StepLR
from io import open
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import corpus_bleu
from torch import optim

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('using device:', device)

if torch.backends.mps.is_available(): # for mac use mps instead of cpu
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('using device:', device)

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15
attention_plot_counter = 1


class Vocab:
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2

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


def split_lines(input_file):
    logging.info("Reading lines of %s...", input_file)
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)
    train_pairs = split_lines(train_file)
    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])
    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)
    return src_vocab, tgt_vocab


def tensor_from_sentence(vocab, sentence):
    indexes = []
    for word in sentence.split():
        if word in vocab.word2index:
            indexes.append(vocab.word2index[word])
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.reduce = nn.Linear(hidden_size * 2, hidden_size) if bidirectional else nn.Identity()

    def forward(self, input_batch, lengths=None):
        if lengths is None:
            if input_batch.dim() == 1:
                input_batch = input_batch.unsqueeze(0)
                lengths = torch.tensor([input_batch.size(1)], device=device)
            elif input_batch.dim() == 2 and input_batch.size(1) == 1:
                input_batch = input_batch.transpose(0, 1)
                lengths = torch.tensor([input_batch.size(1)], device=device)
        embedded = self.embedding(input_batch)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.reduce(outputs)
        return outputs, (hidden, cell)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.3, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, hidden, encoder_outputs):
        h, c = hidden
        emb = self.dropout(self.embedding(input_step))
        h_t = h[-1].unsqueeze(1)
        repeat_h = h_t.expand(-1, encoder_outputs.size(1), -1)
        energy = torch.tanh(self.attn(torch.cat((repeat_h, encoder_outputs), dim=2)))
        attn_scores = self.v(energy).squeeze(2)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((emb, context), dim=2)
        rnn_input = torch.tanh(self.attn_combine(rnn_input))
        output, hidden = self.lstm(rnn_input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden, attn_weights


def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH, beam_size=5, length_penalty_alpha=0.6):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence).unsqueeze(0)
        input_length = torch.tensor([input_tensor.size(1)], device=device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_length)

        h, c = encoder_hidden
        if getattr(encoder, "bidirectional", False):
            num_layers = encoder.lstm.num_layers
            num_directions = 2
            h = h.view(num_layers, num_directions, h.size(1), h.size(2))
            c = c.view(num_layers, num_directions, c.size(1), c.size(2))
            h_cat = torch.cat((h[-1, 0], h[-1, 1]), dim=1).unsqueeze(0)
            c_cat = torch.cat((c[-1, 0], c[-1, 1]), dim=1).unsqueeze(0)
            h = encoder.reduce(h_cat)
            c = encoder.reduce(c_cat)
        else:
            h = h[-1:].contiguous()
            c = c[-1:].contiguous()

        decoder_hidden = (h, c)

        beams = [(0.0, [SOS_index], decoder_hidden, torch.zeros(max_length, input_length.item(), device=device))]
        completed = []

        for t in range(max_length):
            new_beams = []
            for log_prob, seq, hidden, attn_history in beams:
                if seq[-1] == EOS_index:
                    completed.append((log_prob, seq, attn_history))
                    continue
                decoder_input = torch.tensor([[seq[-1]]], device=device)
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, hidden, encoder_outputs)
                log_probs = F.log_softmax(decoder_output, dim=1)
                topv, topi = log_probs.topk(beam_size)
                for i in range(beam_size):
                    next_token = topi[0][i].item()
                    prob = topv[0][i].item()
                    new_seq = seq + [next_token]
                    lp = ((5 + len(new_seq)) / 6) ** length_penalty_alpha
                    new_log_prob = (log_prob + prob) / lp
                    new_attn = attn_history.clone()
                    if decoder_attention is not None:
                        if decoder_attention.dim() == 2:
                            decoder_attention = decoder_attention.squeeze(0)
                        attn_len = min(new_attn.size(1), decoder_attention.size(0))
                        new_attn[len(new_seq) - 2, :attn_len] = decoder_attention[:attn_len]
                    new_beams.append((new_log_prob, new_seq, decoder_hidden, new_attn))
            if not new_beams:
                break
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

        if not completed:
            completed = beams

        best = max(completed, key=lambda x: x[0])
        best_seq = best[1]
        decoded_words = [tgt_vocab.index2word[idx] for idx in best_seq[1:] if idx not in [EOS_index, SOS_index]]
        attentions = best[2]
        return decoded_words, attentions


def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH, beam_size=5, length_penalty_alpha=0.6):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, _ = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab, max_length=max_length, beam_size=beam_size, length_penalty_alpha=length_penalty_alpha)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


def clean(strx):
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=1024, type=int)
    ap.add_argument('--initial_learning_rate', default=0.0003, type=float)
    ap.add_argument('--src_lang', default='fr')
    ap.add_argument('--tgt_lang', default='en')
    ap.add_argument('--train_file', default='data/fren.train.bpe')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe')
    ap.add_argument('--test_file', default='data/fren.test.bpe')
    ap.add_argument('--out_file', default='translations')
    ap.add_argument('--load_checkpoint', nargs=1)
    ap.add_argument('--num_epochs', default=10, type=int)
    ap.add_argument('--beam_size', default=5, type=int)
    ap.add_argument('--length_penalty', default=0.6, type=float)
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0], weights_only=False, map_location=device)
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        src_vocab, tgt_vocab = make_vocabs(args.src_lang, args.tgt_lang, args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=args.initial_learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.85)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])
        # optimizer.load_state_dict(state['opt_state'])

    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    batch_size = 32 # 32
    best_bleu = 0

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

    def train_epoch():
        total_sentences = 0
        total_loss = 0
        for src_batch, src_lens, tgt_batch, tgt_lens in batchify(train_pairs, src_vocab, tgt_vocab, batch_size):
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            optimizer.zero_grad()
            encoder_outputs, encoder_hidden = encoder(src_batch, src_lens)
            h, c = encoder_hidden
            if getattr(encoder, "bidirectional", False):
                num_layers = encoder.lstm.num_layers
                num_directions = 2
                h = h.view(num_layers, num_directions, h.size(1), h.size(2))
                c = c.view(num_layers, num_directions, c.size(1), c.size(2))
                h_cat = torch.cat((h[-1, 0], h[-1, 1]), dim=1).unsqueeze(0)
                c_cat = torch.cat((c[-1, 0], c[-1, 1]), dim=1).unsqueeze(0)
                h = encoder.reduce(h_cat); c = encoder.reduce(c_cat)
            else:
                h = h[-1:].contiguous(); c = c[-1:].contiguous()
            max_tgt_len = tgt_batch.size(1)
            loss = 0
            decoder_input = torch.full((src_batch.size(0), 1), SOS_index, dtype=torch.long, device=device)
            decoder_hidden = (h, c)
            for t in range(max_tgt_len):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, tgt_batch[:, t])
                teacher_force = random.random() < 0.9
                top1 = decoder_output.argmax(1).unsqueeze(1)
                decoder_input = tgt_batch[:, t].unsqueeze(1) if teacher_force else top1
            loss.backward(); optimizer.step()
            total_loss += loss.item() / max_tgt_len
            total_sentences += src_batch.size(0)
        return total_loss / len(train_pairs)

    for epoch in range(args.num_epochs):
        avg_loss = train_epoch()
        print(f"[Epoch {epoch + 1}/{args.num_epochs}] Avg loss: {avg_loss:.4f}")
        scheduler.step()
        dev_translations = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, max_num_sentences=200, beam_size=args.beam_size, length_penalty_alpha=args.length_penalty)
        references = [[clean(pair[1]).split()] for pair in dev_pairs[:200]]
        hypotheses = [sent.split() for sent in dev_translations]
        bleu = corpus_bleu(references, hypotheses)
        print(f"[Epoch {epoch + 1}] Dev BLEU = {bleu:.4f}")
        state = {
            'epoch': epoch,
            'enc_state': encoder.state_dict(),
            'dec_state': decoder.state_dict(),
            'opt_state': optimizer.state_dict(),
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
        }
        torch.save(state, f'state_epoch_{epoch + 1:03d}.pt')
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(state, 'best_model.pt')
            print(f"New best model saved with BLEU {bleu:.4f}")

    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab, beam_size=args.beam_size, length_penalty_alpha=args.length_penalty)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')


if __name__ == '__main__':
    main()