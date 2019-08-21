# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch as t
import numpy as np
import resource

from tqdm import tqdm
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from model import SkipGramNeg

#from data_utils import build_dataset, DataPipeline
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
print("Using {}".format(t.cuda.device_count()))
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 4, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

memory_limit()

#available, total = cuda.mem_get_info()
#print("Available: %.2f GB\nTotal:     %.2f GB"%(available/1e9, total/1e9))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=200, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=10, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=5, help="number of epochs")
    parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    return parser.parse_args()


class PermutedSubsampledCorpus(Dataset):
    def __init__(self, datapath, data_utils, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data
        #self.window_size = window_size
        #self.batch_neg = get_neg_data(args.mb, args.n_negs, self.data[1])
        self.iword, self.owords = list(zip(*data))
        self.owords = t.tensor([random.sample(oword, 1) for oword in self.owords], dtype=t.long).squeeze()
        #self.vocab = vocab
        self.data_utils = data_utils

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #iword, owords = self.data[idx]
        #sample = {'i':iword, 'o':np.array(owords)}
        #print(self.iword[idx])
        #print(self.owords[idx])
        #boundary = np.random.randint(1, self.window_size)
        return self.iword[idx], self.owords[idx], self.data_utils.getNegatives(self.owords[idx])
        #return self.iword[idx], self.owords[idx]  #sample

class DataUtils:
    def __init__(self, wc, n_negs):
        self.wc = wc
        self.negative_table_size = 1e8
        self.negatives = []
        self.negpos = 0
        self.n_negs = n_negs

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.wc.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * self.negative_table_size)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target):  # TODO check equality with target
        size = self.n_negs
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            response = np.concatenate((response, self.negatives[0:self.negpos]))
        while target.tolist() in response:
            #print("already seen")
            self.negpos = (self.negpos + size) % len(self.negatives)
            response = self.negatives[self.negpos:self.negpos + size]
            if len(response) != size:
                response = np.concatenate((response, self.negatives[0:self.negpos]))
        return response

def get_neg_data(batch_size, num, target_inputs, vocab):
    neg = np.zeros(num)
    print(len(target_inputs))
    for j in range(len(target_inputs)):
        print(j)
        for i in range(len(target_inputs[j])):
            delta = random.sample(vocab, num)
            #print(target_inputs[i])
            #print(delta)
            #print(target_inputs[j][i])
            while target_inputs[j][i] in delta:
                print("sample new")
                delta = random.sample(target_inputs[j-1], num)
            neg = np.vstack([neg, delta])
            print(neg.shape)
    return neg[1:batch_size+1]

def train(args):
    #print(t.get_num_threads())
    t.set_num_threads(8)
    print("Number of threads: {}".format(t.get_num_threads()))
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(args.data_dir, 'word2idx.dat'), 'rb'))
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    # frequency subsampling
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    weights = wf if args.weights else None
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    model = SkipGramNeg(vocab_size=vocab_size, emb_dim=args.e_dim)
    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))
    #sgns = SGNS(embedding=model, vocab=word2idx.keys(), vocab_size=vocab_size, n_negs=args.n_negs, weights=weights)
    if os.path.isfile(modelpath) and args.conti:
        model.load_state_dict(t.load(modelpath))
    if args.cuda:
        model = model.cuda()
    optim = SGD(model.parameters(), lr=0.01)
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and args.conti:
        optim.load_state_dict(t.load(optimpath))
    data_utils = DataUtils(wc, args.n_negs)
    data_utils.initTableNegatives()
    dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'), data_utils)
    #pipeline = DataPipeline(dataset, range(len(idx2word)), vocab_size, data_offest=0, use_noise_neg=False)
    #vali_examples = random.sample(word2idx.keys(), vali_size)
    for epoch in range(1, args.epoch + 1):
        batch_size = len(dataset)//args.mb
        #print(batch_size)
        #print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        
        #batch_inputs, batch_labels = pipeline.generate_batch(batch_size, num_skips, skip_window)
        #batch_neg = pipeline.get_neg_data(batch_size, num_neg, batch_inputs)
        total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        #print(len(list(map(lambda item: item[1].tolist(), dataloader))[0]))
        #batch_neg = get_neg_data(batch_size, args.n_negs, list(map(lambda item: item[1].tolist(), dataloader)), range(len(idx2word)))
        #batch_neg = t.tensor(batch_neg, dtype=t.long)
        for iword, owords, batch_neg in pbar:
            iword = iword.to(device)
            owords = owords.to(device)
            batch_neg = batch_neg.to(device)
            
            loss = model(iword, owords, batch_neg)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
            print("Loss: {}".format(loss))
    idx2vec = model.input_emb.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(args.data_dir, 'idx2vec.dat'), 'wb'))
    t.save(model.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))

if __name__ == '__main__':
    train(parse_args())
