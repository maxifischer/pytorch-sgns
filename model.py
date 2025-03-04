import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGramNeg, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim, sparse=True).to(device)
        #print(self.input_emb.size())
        self.output_emb = nn.Embedding(vocab_size, emb_dim, sparse=True).to(device)
        #print(self.output_emb.size())
        self.log_sigmoid = nn.LogSigmoid()

        initrange = (2.0 / (vocab_size + emb_dim)) ** 0.5  # Xavier init
        self.input_emb.weight.data.uniform_(-initrange, initrange)
        #print(self.input_emb.size())
        self.output_emb.weight.data.uniform_(-0, 0)
        #print(self.output_emb.size())

    def forward(self, target_input, context, neg):
        """
        :param target_input: [batch_size]
        :param context: [batch_size]
        :param neg: [batch_size, neg_size]
        :return:
        """
        # u,v: [batch_size, emb_dim]
        v = self.input_emb(target_input)
        u = self.output_emb(context)
        # positive_val: [batch_size]
        #print(u.size())
        #print(v.size())
        positive_val = self.log_sigmoid(torch.sum(u * v, dim=1)).squeeze()
        #print(positive_val.size())
        # u_hat: [batch_size, neg_size, emb_dim]
        #print(neg.size())
        u_hat = self.output_emb(neg)
        # [batch_size, neg_size, emb_dim] x [batch_size, emb_dim, 1] = [batch_size, neg_size, 1]
        # neg_vals: [batch_size, neg_size]
        #print(u_hat.size())
        #print(v.size())
        neg_vals = torch.bmm(u_hat, v.unsqueeze(2)).squeeze(2)
        # neg_val: [batch_size]
        neg_val = self.log_sigmoid(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = positive_val + neg_val
        return -loss.mean()

    def predict(self, inputs):
        return self.input_emb(inputs)
