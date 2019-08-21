# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, window=2, unk='<UNK>', data_dir='./data/'):
        self.window = window
        self.unk = unk
        self.data_dir = data_dir

    def process_file(self, filepath):
        #print("Processing file...")
        with codecs.open(filepath, 'r', encoding='utf-8') as loaded_file:
            corpus = []
            #print("Loaded file...")
            text = loaded_file.read()
            #print(len(text))
            text = text.lower()
            #print(len(text))
            text.replace('\n', ' ')
            #print(len(text))
            text = re.sub('[^a-z ]+', '', text)
            #print(len(text))
            corpus.extend([w for  w in text.split() if w != ''])
            #print(len(corpus))
        self.text = corpus

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        #print("iword: {}".format(iword))
        #print("owords: {}".format(right + [self.unk for _ in range(self.window - len(right))]))
        return iword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self, filepath, max_vocab=20000):
        print("building vocab...")
        step = 0
        self.wc = {self.unk: 1}
        self.tc = 0
        #with codecs.open(filepath, 'r', encoding='utf-8') as file:
        #    for line in file:
        #        step += 1
        #        if not step % 1000:
        #            print("working on {}kth line".format(step // 1000), end='\r')
        #        line = line.strip()
        #        if not line:
        #            continue
        #        num_appearances = int(line.split()[-1])
        #        print(num_appearances)
        #        data_nonumbers = ''.join(c for c in line if not c.isdigit())
        #        sent = data_nonumbers.split()
        for word in self.text:
            self.tc += 1
            self.wc[word] = self.wc.get(word, 0) + 1
        self.idx2word = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        print("Vocabulary size: {}".format(len(self.vocab)))
        pickle.dump(self.tc, open(os.path.join(self.data_dir, 'tc.dat'), 'wb'))
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'wc.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        print("build done")

    def convert(self, filepath):
        print("converting corpus...")
        step = 0
        data = []
        no_input = 0
        no_target = 0
        #self.text = self.preprocess_file(filepath)
        sent = []
        for word in self.text:
            if word in self.vocab:
                sent.append(word)
            else:
                sent.append(self.unk)
                #for i in range(len(sent)):
                #print(sent)
        #if iword=="<UNK>":
                    #print(iword)
            #        no_input+=1
            #        continue
            #    elif len(set(owords))==1:
                    #print(owords)
            #        no_target+=1
            #        continue
            #    else:
        for i in range(len(sent)):
            iword, owords = self.skipgram(sent, i)
            #print(iword)
            #print(owords)
            data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
        #print("{0}x no input, {1}x no target".format(no_input, no_target))
        print(len(data))
        print(data[:5])
        pickle.dump(data, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        print("conversion done")


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    preprocess.process_file(args.corpus)
    #print(text[:20])
    print(len(preprocess.text))
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.corpus)
