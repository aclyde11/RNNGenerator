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

def getconfig(args):
    return args

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
    with open(fname, 'r') as f:
        lines1 = []
        lines2 = []
        for y in tqdm(map(lambda x: x.split(','), (filter(lambda x: len(x) != 0, map(lambda x: x.strip(), f))))):
            maps = list(map(lambda x: int(x), y))
            lines1.append(torch.from_numpy(np.array([c2i(START_CHAR)] + maps, dtype=np.int64)))
            lines2.append(torch.from_numpy(np.array(maps + [c2i(END_CHAR)], dtype=np.int64)))
        print("Read", len(lines2), "SMILES.")

    return lines1, lines2



def sample(model, i2c, c2i, device, z_dim=2, temp=1, batch_size=10, max_len=320):
    model.eval()
    with torch.no_grad():

        c_0 = torch.zeros((4, batch_size, 256)).to(device)
        h_0 = torch.zeros((4, batch_size, 256)).to(device)

        x = torch.tensor(c2i(START_CHAR)).unsqueeze(0).unsqueeze(0).repeat((max_len, batch_size)).to(device)

        eos_mask = torch.zeros(batch_size, dtype=torch.bool).to(device)
        end_pads = torch.tensor([max_len - 1]).repeat(batch_size).to(device)
        for i in range(1, max_len):
            x_emb = model.emb(x[i - 1, :]).unsqueeze(0)
            o, (h_0, c_0) = model.lstm(x_emb, (h_0, c_0))
            # o, h_0 = model.lstm(x_emb, h_0)
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


def train_epoch(model, optimizer, dataloader, args, device):
    model.train()
    lossf = nn.CrossEntropyLoss().to(device)
    losses = []
    counters =0
    for i, (y, y_hat) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        y = [x.to(device) for x in y]
        batch_size = len(y)
        packed_seq_hat, _ = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.pack_sequence(y_hat, enforce_sorted=False),
                                                             total_length=args.maxlen)
        pred = model(y)
        packed_seq_hat = packed_seq_hat.view(-1).long()
        pred = pred.view(batch_size * args.maxlen, -1)
        loss = lossf(pred, packed_seq_hat.to(device)).mean()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    return np.array(losses).flatten().mean()


def main(args, device):
    args = getconfig(args)
    print("loading data.")
    vocab, c2i, i2c, _, _ = get_vocab_from_file(args.i + "/vocab.txt")
    print("Vocab size is", len(vocab))
    s, e = get_input_data(args.i + "/out.txt", c2i)
    input_data = ToyDataset(s, e)
    print("Done.")

    ## make data generator
    dataloader = torch.utils.data.DataLoader(input_data, pin_memory=True, batch_size=args.b,
                                             collate_fn=mycollate)

    model = CharRNN(len(vocab), len(vocab), max_len=args.maxlen).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epoch_start = 0
    if args.ct:
        print("Continuing from save.")
        pt = torch.load(args.logdir + "/autosave.model.pt")
        model.load_state_dict(pt['state_dict'])
        optimizer.load_state_dict(pt['optim_state_dict'])
        epoch_start = pt['epoch'] + 1

    with open(args.logdir + "/training_log.csv", 'w') as flog:
        if args.e is None:
            flog.write("epoch,train_loss,sampled,valid")
            for epoch in range(epoch_start, args.e):
                avg_loss = train_epoch(model, optimizer, dataloader, args, device)
                samples = sample(model, i2c, c2i, device, batch_size=args.b , max_len=args.maxlen)
                valid = count_valid_samples(samples)
                print(samples)
                print("Total valid samples:", valid, float(valid) / 1024)
                flog.write( ",".join([str(epoch), str(avg_loss), str(len(samples)), str(valid)]) + "\n")
                torch.save(
                    {
                        'state_dict' : model.state_dict(),
                        'optim_state_dict' : optimizer.state_dict(),
                        'epoch' : epoch
                    }, args.logdir + "/autosave.model.pt"
                )
        else:
            flog.write("epoch,train_loss,sampled,valid")
            for epoch in range(epoch_start, epoch_start + args.e):
                avg_loss = train_epoch(model, optimizer, dataloader, args, device)
                samples = sample(model, i2c, c2i, device, batch_size=args.b, max_len=args.maxlen)
                valid = count_valid_samples(samples)
                print(samples)
                print("Total valid samples:", valid, float(valid) / 1024)
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
    parser.add_argument('-b', help='batch size', type=int, default=256)
    parser.add_argument('--logdir', help='place to store things.', type=str, required=True)
    parser.add_argument('--ct', help='continue training for longer',action='store_true')
    parser.add_argument('-e', type=int, required=False, default=None)
    parser.add_argument('--maxlen', type=int, required=True, default=None)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("Device: ", device)
    path = args.logdir
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed. Maybe it already exists? I will overwrite :)" % path)
    else:
        print("Successfully created the directory %s " % path)

    main(args, device)
