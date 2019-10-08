import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class VAERNN(nn.Module):
    def __init__(self, vocab_size, emb_size, max_len=150):
        super(VAERNN, self).__init__()
        self.encoder = EncoderCharRNN(vocab_size, emb_size, max_len)
        self.decoder = CharRNN(vocab_size, emb_size, max_len)

    def forward(self,x):
        posteriors  = self.encoder(x)

    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.Variable(torch.randn(mb_size, Z_dim))
        return mu + torch.exp(log_var / 2) * eps

class EncoderCharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, max_len=150):
        super(CharRNN, self).__init__()
        self.max_len = max_len
        self.emb  = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, 256, dropout=0.3, num_layers=2)
        self.linear = nn.Linear(256, 2)

    # pass x as a pack padded sequence please.
    def forward(self, x, with_softmax=False):
        # do stuff to train
        dv = x[0].device
        xs = []
        for x_ in x:
            x_ = x_.cpu().numpy()
            x_ = np.flip(x_, axis=-1)
            x_ = x_.copy()
            x_ = self.emb(torch.from_numpy(x_).to(dv))
            xs.append(x_)

        x = xs
        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

        x,_ = self.lstm(x)

        x, lens  = nn.utils.rnn.pad_packed_sequence(x, padding_value=0, total_length=self.max_len)
        x = self.linear(x)
        if with_softmax:
            return F.softmax(x, dim=-1)
        else:
            return x

    def sample(self):
        return None

#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836943/pdf/MINF-37-na.pdf
class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, max_len=150):
        super(CharRNN, self).__init__()
        self.max_len = max_len
        self.emb  = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, 256, dropout=0.3, num_layers=2)
        self.linear = nn.Linear(256, vocab_size)

    # pass x as a pack padded sequence please.
    def forward(self, x, with_softmax=False):
        # do stuff to train
        x = [self.emb(x_) for x_ in x]

        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

        x,_ = self.lstm(x)

        x, _  = nn.utils.rnn.pad_packed_sequence(x, padding_value=0, total_length=self.max_len)
        print(x.shape)
        x = self.linear(x)
        if with_softmax:
            return F.softmax(x, dim=-1)
        else:
            return x

    def sample(self):
        return None
