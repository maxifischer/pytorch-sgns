import argparse
import os
import pickle
import torch as t
from model import SkipGramNeg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../ChronologicalTrainedSkipGram/data/pytorch_data/')
    parser.add_argument('--vocab_size', type=int, default=42626)
    parser.add_argument('--e_dim', type=int, default=200)
    parser.add_argument('--n_negs', type=int, default=20)
    return parser.parse_args()

def most_similar(args, word, top_k=8):
    print("The most similar words to {} are:".format(word))
    word2idx = pickle.load(open(os.path.join(args.data_dir, 'word2idx.dat'), 'rb'))
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    #model = Word2Vec(vocab_size=args.vocab_size, embedding_size=args.e_dim)
    #model = SGNS(embedding=model, vocab=word2idx.keys(), vocab_size=args.vocab_size, n_negs=args.n_negs, weights=None)
    model = SkipGramNeg(vocab_size=args.vocab_size, emb_dim=args.e_dim)
    model.load_state_dict(t.load("./pts/sgns.pt"))
    model.eval()

    #print("Model's state_dict:")
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    index = word2idx[word]
    index = t.tensor(index, dtype=t.long).cpu().unsqueeze(0)
    emb = model.predict(index)
    #print(emb)
    sim = t.mm(emb, model.output_emb.weight.transpose(0, 1))
    nearest = (-sim[0]).sort()[1][1: top_k + 1]
    top_list = []
    for k in range(top_k):
        close_word = idx2word[nearest[k].item()]
        top_list.append(close_word)
    return top_list

if __name__ == '__main__':
    args = parse_args()
    print(most_similar(args, 'gay'))
    print(most_similar(args, 'apple'))
    print(most_similar(args, 'cat'))
    print(most_similar(args, 'france'))
