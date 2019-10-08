import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class VAERNN(nn.Module):
    def __init__(self, vocab_size, emb_size, z_size=4, max_len=150):
        super(VAERNN, self).__init__()
        self.encoder = EncoderCharRNN(vocab_size, emb_size, z_size, max_len)
        self.decoder = DecoderCharRNN(vocab_size, emb_size, z_size, max_len)

    def forward(self,x, return_mu=True):
        mu, logvar, x_padded, lens  = self.encoder(x)
        z = self.sample_z(mu, logvar)
        x = self.decoder(x_padded,z)
        if return_mu:
            return x, (mu, logvar)
        else:
            return x



    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.autograd.Variable(torch.randn(mu.shape)).to(mu.device)
        return mu + torch.exp(log_var / 2.0) * eps

class EncoderCharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, z_dim, max_len=150):
        super(EncoderCharRNN, self).__init__()
        self.max_len = max_len
        self.emb  = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, 256, bidirectional=True, dropout=0.1, num_layers=2)
        self.linear1 = nn.Linear(256, z_dim)
        self.linear2 = nn.Linear(256, z_dim)

    # pass x as a pack padded sequence please.
    def forward(self, x):
        # do stuff to train
        x = [self.emb(x_) for x_ in x]
        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        x_padded, lens  = nn.utils.rnn.pad_packed_sequence(x, padding_value=0, total_length=self.max_len)

        x,_ = self.lstm(x_padded)

        x = x.view(x.shape[0], x.shape[1], 2, -1)[:,:,1,:] # get backwards only.

        mu = self.linear1(x)
        logvar = self.linear2(x)
        #   x = [max_len, batch_size, mu]
        #   x = [max_len, batch_size, sigma]

        return mu, logvar, x_padded, lens

    def sample(self):
        return None

class DecoderCharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, z_size, max_len=150):
        super(DecoderCharRNN, self).__init__()
        self.max_len = max_len
        self.lstm = nn.LSTM(z_size + emb_size, 256, dropout=0.3, num_layers=2)
        self.linear = nn.Linear(256, vocab_size)
        self.dropout = nn.Dropout(0.1)

    # pass x as a pack padded sequence please.
    def forward(self, x, z, with_softmax=False):
        # do stuff to train
        x = torch.cat([self.dropout(x),z], dim=-1)
        x,_ = self.lstm(x)

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

        x = self.linear(x)
        if with_softmax:
            return F.softmax(x, dim=-1)
        else:
            return x

    def sample(self):
        return None
