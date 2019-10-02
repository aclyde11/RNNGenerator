import argparse
from model.vocab import get_vocab_from_file, START_CHAR, END_CHAR
from model.model import CharRNN
import torch.utils.data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn
import torch.nn.functional as F
from tqdm import tqdm
import os

from train import getconfig, get_vocab_from_file,


def count_valid_samples(smiles):
    from rdkit import Chem
    count = 0
    goods = []
    for smi in smiles:
        try:
            mol     = Chem.MolFromSmiles(smi[1:-1])
            goods.append(Chem.MolToSmiles(mol))
        except:
            continue
        if mol is not None:
            count += 1
    return count, goods


def sample(model, i2c, c2i, device, temp=1, batch_size=10, max_len=150):
    model.eval()
    with torch.no_grad():

        c_0 = torch.zeros((2, batch_size, 256)).to(device)
        h_0 = torch.zeros((2, batch_size, 256)).to(device)
        x = torch.tensor(c2i(START_CHAR)).unsqueeze(0).unsqueeze(0).repeat((max_len, batch_size)).to(device)

        eos_mask = torch.zeros(batch_size, dtype=torch.bool).to(device)
        end_pads = torch.tensor([max_len - 1]).repeat(batch_size).to(device)
        for i in range(1, max_len):
            x_emb = model.emb(x[i - 1, :]).unsqueeze(0)
            o, (h_0, c_0) = model.lstm(x_emb, (h_0, c_0))
            y = model.linear(o.squeeze(0))
            y = F.softmax(y / temp, dim=-1)
            w = torch.multinomial(y, 1).squeeze()
            x[i, ~eos_mask] = w[~eos_mask]

            i_eos_mask = ~eos_mask & (w == c2i(END_CHAR))
            end_pads[i_eos_mask] = i + 1
            eos_mask = eos_mask | i_eos_mask

        new_x = []
        for i in range(x.size(1)):
            new_x.append(x[:end_pads[i], i].cpu())
        return ["".join(map(i2c, list(i_x.cpu().flatten().numpy()))) for i_x in new_x]


def main(args, device):
    config = getconfig(args)
    print("loading data.")
    vocab, c2i, i2c = get_vocab_from_file(args.i + "/vocab.txt")


    model = CharRNN(config['vocab_size'], config['emb_size'], max_len=config['max_len']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pt = torch.load(args.logdir + "/autosave.model.pt", map_location=device)
    model.load_state_dict(pt['state_dict'])
    optimizer.load_state_dict(pt['optim_state_dict'])

    total_sampled = 0
    total_valid = 0
    total_unqiue = 0
    smiles = set()
    for epoch in range(args.n / 512):
        samples = sample(model, i2c, c2i, device, batch_size=512, max_len=config['max_len'])
        samples = list(map(lambda x : x[1:-1], samples))
        total_sampled += len(samples)
        if args.v:
            valid_smiles, goods = count_valid_samples(samples)
            total_valid += total_valid
            smiles.update(valid_smiles)
        else:
            smiles.update(samples)
    smiles = list(smiles)
    total_unqiue += len(smiles)

    with open(args.o, 'w') as f:
        for i in smiles:
            f.write(i)
            f.write('\n')

    print("output smiles to", args.o)
    print("Sampled", total_sampled)
    print("Total unique", total_unqiue, float(total_unqiue)/float(total_sampled))
    if args.v:
        print("total valid", total_valid, float(total_valid)/float(total_valid))



if __name__ == '__main__':
    print("Note: This script is very picky. This will only run on a CPU at this time, but will be extended to GPU. ")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Data from vocab folder', type=str, required=True)
    parser.add_argument('--logdir', help='place to store things.', type=str, required=True)
    parser.add_argument('-o', required=True, help='place to store output smiles', type=str)
    parser.add_argument('-n', help='number samples to test', type=int, required=True)
    parser.add_argument('-v', help='validate, uses rdkit', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    main(args, device)
