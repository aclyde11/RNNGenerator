import argparse
from model.vocab import get_vocab_from_file, START_CHAR, END_CHAR
from model.model import VAERNN
import torch.utils.data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn
import torch.nn.functional as F
from tqdm import tqdm
import os

def getconfig(args):
    config_ = {
        'epochs': 220,
        'batch_size': 128,
        'vocab_size': 28,
        'emb_size': 32,
        'sample_freq': 1,
        'max_len': 180,
        'z_size' : 6
    }

    return config_

def count_valid_samples(smiles):
    from rdkit import Chem
    count = 0
    for smi in smiles:
        try:
            mol     = Chem.MolFromSmiles(smi[1:-1])
        except:
            continue
        if mol is not None:
            count += 1
    return count

def get_input_data(fname, c2i):
    lines = open(fname, 'r').readlines()[:250000]
    lines = list(map(lambda x: x.split(','), (filter(lambda x: len(x) != 0, map(lambda x: x.strip(), lines)))))

    lines1 = [torch.from_numpy(np.array([c2i(START_CHAR)] + list(map(lambda x: int(x), y)), dtype=np.int64)) for y in
              lines]
    lines2 = [torch.from_numpy(np.array(list(map(lambda x: int(x), y)) + [c2i(END_CHAR)], dtype=np.int64)) for y in
              lines]
    print("Read", len(lines2), "SMILES.")

    return lines1, lines2


def sample(model, i2c, c2i, device, z_dim=2, temp=1, batch_size=10, max_len=150, alpha=0.2, num_layers=2):
    model.eval()
    with torch.no_grad():

        h = (torch.zeros((num_layers, batch_size, 256)).to(device), torch.zeros((num_layers, batch_size, 256)).to(device))
        x = torch.tensor(c2i(START_CHAR)).unsqueeze(0).unsqueeze(0).repeat((max_len, batch_size)).to(device)

        z = torch.randn((1, batch_size, z_dim)).to(device)

        eos_mask = torch.zeros(batch_size, dtype=torch.bool).to(device)
        end_pads = torch.tensor([max_len - 1]).repeat(batch_size).to(device)
        for i in range(1, max_len):
            x_emb = model.encoder.emb(x[i - 1, :]).unsqueeze(0)

            z = alpha * z + torch.randn(z.shape, device=z.device) * (1-(alpha * alpha)) + 0.0 #AR

            x_emb = torch.cat([x_emb, z], dim=-1)
            o, h = model.decoder.lstm(x_emb, (h))
            y = model.decoder.linear(o.squeeze(0))
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


def mycollate(x):
    x_batches = []
    y_batchese = []
    for i in x:
        x_batches.append(i[0])
        y_batchese.append(i[1])
    return x_batches, y_batchese


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, s, e):
        self.s = s
        self.e = e
        assert (len(self.s) == len(self.e))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, item):
        return self.s[item], self.e[item]


def train_epoch(model, optimizer, dataloader, config, device, epoch=1):
    model.train()
    lossf = nn.CrossEntropyLoss().to(device)
    losses = []
    beta = ((1e-3) / float(config['max_len']) ) * (5 * epoch)
    iters = tqdm(enumerate(dataloader), postfix={'loss' : 0, 'kl' : 0})
    for i, (y, y_hat) in iters:
        optimizer.zero_grad()

        y = [x.to(device) for x in y]
        batch_size = len(y)
        packed_seq_hat, _ = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.pack_sequence(y_hat, enforce_sorted=False),
                                                             total_length=config['max_len'])
        pred, (mu, logvar) = model(y, return_mu=True, force=(epoch < 5), prob_forcing=max(0, 1.0 - (epoch * 0.05)))
        packed_seq_hat = packed_seq_hat.view(-1).long()
        pred = pred.view(batch_size * config['max_len'], -1)
        loss = lossf(pred, packed_seq_hat.to(device)).mean()

        kldiv = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss += beta * kldiv
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if i % 2 == 0:
            iters.set_postfix({'loss' : loss.item(), 'kl' : kldiv.item()})


    return np.array(losses).flatten().mean()


def main(args, device):
    config = getconfig(args)
    print("loading data.")
    vocab, c2i, i2c = get_vocab_from_file(args.i + "/vocab.txt")
    print("Vocab size is", len(vocab))
    s, e = get_input_data(args.i + "/out.txt", c2i)
    input_data = ToyDataset(s, e)
    print("Done.")

    ## make data generator
    dataloader = torch.utils.data.DataLoader(input_data, pin_memory=True, batch_size=config['batch_size'],
                                             collate_fn=mycollate)

    model = VAERNN(config['vocab_size'], config['emb_size'], startchar=c2i(START_CHAR), endchar=c2i(END_CHAR), z_size=config['z_size'], max_len=config['max_len']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epoch_start = 0
    if args.ct:
        print("Continuing from save.")
        pt = torch.load(args.logdir + "/autosave.model.pt")
        model.load_state_dict(pt['state_dict'])
        optimizer.load_state_dict(pt['optim_state_dict'])
        epoch_start = pt['epoch'] + 1

    with open(args.logdir + "/training_log.csv", 'w') as flog:
        flog.write("epoch,train_loss,sampled,valid")
        for epoch in range(epoch_start, config['epochs']):
            avg_loss = train_epoch(model, optimizer, dataloader, config, device, epoch=epoch)
            samples = sample(model, i2c, c2i, device, config['z_size'], batch_size=8, max_len=config['max_len'])
            valid = count_valid_samples(samples)
            print(samples)
            print("Total valid samples:", valid, float(valid))
            flog.write( ",".join([str(epoch), str(avg_loss), str(len(samples)), str(valid)]) + "\n")
            torch.save(
                {
                    'state_dict' : model.state_dict(),
                    'optim_state_dict' : optimizer.state_dict(),
                    'epoch' : epoch
                }, args.logdir + "/autosave.model.pt"
            )


if __name__ == '__main__':
    print("Note: This script is very picky. This will only run on a GPU. ")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Data from vocab folder', type=str, required=True)
    parser.add_argument('--logdir', help='place to store things.', type=str, required=True)
    parser.add_argument('--ct', help='continue training for longer', type=bool, default=False)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    path = args.logdir

    import torch
    torch.autograd.set_detect_anomaly(True)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed. Maybe it already exists? I will overwrite :)" % path)
    else:
        print("Successfully created the directory %s " % path)

    main(args, device)
