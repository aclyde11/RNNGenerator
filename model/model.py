import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from model.vocab import END_CHAR
import numpy as np


class VAERNN(nn.Module):
    def __init__(self, vocab_size, emb_size, z_size=4, max_len=150, endchar=34, startchar=23):
        super(VAERNN, self).__init__()
        self.encoder = EncoderCharRNN(vocab_size, emb_size, z_size, max_len)
        self.decoder = DecoderCharRNN(vocab_size, emb_size, z_size, max_len)
        self.endchar = endchar
        self.startchar = startchar

    def forward(self,x, return_mu=True, prob_forcing=0.5, force=True):
        mu, logvar, x_padded, lens  = self.encoder(x)
        z = self.sample_z(mu, logvar)
        x = self.decoder(x_padded,z, self.endchar, self.startchar, self.encoder.emb, force=True, prob_forcing=prob_forcing)
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
        self.emb.weight.requires_grad = False
        self.lstm = nn.RNN(emb_size, 256, bidirectional=True, dropout=0, num_layers=2)
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


        return mu, logvar, x_padded, lens

    def sample(self):
        return None

class DecoderCharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, z_size, max_len=150):
        super(DecoderCharRNN, self).__init__()
        self.max_len = max_len

        self.vocab_size = vocab_size
        self.num_layers= 2
        self.lstm = nn.GRU(z_size, 256, dropout=0.1, num_layers=self.num_layers)
        self.linear = nn.Linear(256, vocab_size)
        self.dropout = nn.Dropout(0.1)

    # pass x as a pack padded sequence please.
    def forward(self, x_actual, z, endchar, startchar, ember, force=True, prob_forcing=0.25, with_softmax=False):
        # do stuff to train

        # if force:
        x, _ = self.lstm(z)
        x = self.linear(x)
        if with_softmax:
            return F.softmax(x, dim=-1)
        else:
            return x

        # else:
        #     dv = z.device
        #     batch_size = z.shape[1]
        #     h = torch.autograd.Variable(torch.zeros((self.num_layers, batch_size, 256)).to(dv)) #, torch.autograd.Variable(torch.zeros((self.num_layers, batch_size, 256)).to(dv)))
        #     x = torch.autograd.Variable(torch.tensor(startchar)).unsqueeze(0).unsqueeze(0).repeat((self.max_len, batch_size)).to(dv)
        #     x_res = torch.autograd.Variable(torch.zeros((x_actual.shape[0], x_actual.shape[1], self.vocab_size))).to(dv)
        #     for i in range(1, self.max_len):
        #         if i == 1 or (random.random() < prob_forcing):
        #             x_emb = (x_actual[i - 1, :]).unsqueeze(0)
        #         else:
        #             x_emb = ember(x[i - 1, :]).unsqueeze(0)
        #
        #         x_emb = torch.cat([x_emb, z[i].unsqueeze(0)], dim=-1)
        #         o, h = self.lstm(x_emb, (h))
        #         y = self.linear(o.squeeze(0))
        #         x_res[i] = y
        #         y = F.softmax(y / 1.0, dim=-1)
        #         w = torch.argmax(y, dim=-1).squeeze()
        #         x[i] = w

        # if force:
        #     x, _ = self.lstm(torch.cat([x_actual, z], dim=-1))
        #
        #     x = self.linear(x)
        #     if with_softmax:
        #         return F.softmax(x, dim=-1)
        #     else:
        #         return x
        #
        # else:
        #     dv = z.device
        #     batch_size = z.shape[1]
        #     h = torch.autograd.Variable(torch.zeros((self.num_layers, batch_size, 256)).to(dv)) #, torch.autograd.Variable(torch.zeros((self.num_layers, batch_size, 256)).to(dv)))
        #     x = torch.autograd.Variable(torch.tensor(startchar)).unsqueeze(0).unsqueeze(0).repeat((self.max_len, batch_size)).to(dv)
        #
        #     x_res = torch.autograd.Variable(torch.zeros((x_actual.shape[0], x_actual.shape[1], self.vocab_size))).to(dv)
        #     for i in range(1, self.max_len):
        #         if i == 1 or (random.random() < prob_forcing):
        #             x_emb = (x_actual[i - 1, :]).unsqueeze(0)
        #         else:
        #             x_emb = ember(x[i - 1, :]).unsqueeze(0)
        #
        #         x_emb = torch.cat([x_emb, z[i].unsqueeze(0)], dim=-1)
        #         o, h = self.lstm(x_emb, (h))
        #         y = self.linear(o.squeeze(0))
        #         x_res[i] = y
        #         y = F.softmax(y / 1.0, dim=-1)
        #         w = torch.argmax(y, dim=-1).squeeze()
        #         x[i] = w
        #
        #     if with_softmax:
        #         return F.softmax(x_res, dim=-1)
        #     else:
        #         return x_res

    def sample(self):
        return None


#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836943/pdf/MINF-37-na.pdf
class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, max_len=320):
        super(CharRNN, self).__init__()
        self.max_len = max_len
        self.emb  = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, 256, dropout=0.1, num_layers=3, bidirectional=True)
        self.linear = nn.Linear(256 * 2, vocab_size)

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
