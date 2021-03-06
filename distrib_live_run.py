import multiprocessing

import argparse
import time

import torch
import torch.nn.functional as F
import torch.nn.utils.rnn
import torch.utils.data

from model.model import CharRNN
from model.vocab import START_CHAR, END_CHAR
from train import getconfig, get_vocab_from_file


def count_valid_samples(smiles, rdkit=True):
    if rdkit:
        from rdkit import Chem
        from rdkit import RDLogger
        lg = RDLogger.logger()

        lg.setLevel(RDLogger.CRITICAL)
        def toMol(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                return Chem.MolToSmiles(mol)
            except:
                return None
    else:
        import pybel
        def toMol(smi):
            try:
                m = pybel.readstring("smi", smi)
                return m.write("smi")
            except:
                return None

    count = 0
    goods = []
    for smi in smiles:
        try:
            mol = toMol(smi)
            if mol is not None:
                goods.append(mol)
                count += 1
        except:
            continue
    return count, goods


def sample(model, i2c, c2i, device, temp=1, batch_size=10, max_len=150):

    model.eval()
    with torch.no_grad():

        c_0 = torch.zeros((2, batch_size, 256)).cuda(device)
        h_0 = torch.zeros((2, batch_size, 256)).cuda(device)
        x = torch.tensor(c2i(START_CHAR)).unsqueeze(0).unsqueeze(0).repeat((max_len, batch_size)).cuda(device)

        eos_mask = torch.zeros(batch_size, dtype=torch.bool).cuda(device)
        end_pads = torch.tensor([max_len - 1]).repeat(batch_size).cuda(device)

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

        x = x.cpu().numpy()


        return x, end_pads.cpu()


def main(args, device, queue):
    config = getconfig(args)
    vocab, c2i, i2c = get_vocab_from_file(args.i + "/vocab.txt")

    model = CharRNN(config['vocab_size'], config['emb_size'], max_len=config['max_len']).cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pt = torch.load(args.logdir + "/autosave.model.pt", map_location='cuda:' + str(device))
    model.load_state_dict(pt['state_dict'])
    optimizer.load_state_dict(pt['optim_state_dict'])


    batch_size = args.batch_size if args.batch_size > 1 else config['batch_size']

    for epoch in range(int(args.n / batch_size)):
        samples = sample(model, i2c, c2i, device, batch_size=batch_size, max_len=config['max_len'], temp=args.t)
        queue.put(samples)

def poolProc(inqueue, outqueue, i2c):
    while(True):
        x, end_pads = inqueue.get()
        new_x = []
        for i in range(x.shape[1]):
            jers = x[1:end_pads[i]-1, i]
            new_x.append("".join(map(i2c, jers)))
        outqueue.put(new_x)

def writer(outqueue, total):
    counter = 0
    while counter < total:
        samples = outqueue.get()
        for i in samples:
            print(i)
            counter += 1
    exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Data from vocab folder', type=str, required=True)
    parser.add_argument('--logdir', help='place to store things.', type=str, required=True)
    parser.add_argument('-n', help='number samples to test', type=int, required=True)
    parser.add_argument('-vr', help='validate, uses rdkit', action='store_true')
    parser.add_argument('-vb', help='validate, uses openababel', action='store_true')
    parser.add_argument('-t', help='temperature', default=1.0, required=False, type=float)
    parser.add_argument('--batch_size', default=-1, required=False, type=int)
    args = parser.parse_args()

    gpus = 6
    nworkers = 2 * 6

    resqueue = multiprocessing.Queue()
    outqueue = multiprocessing.Queue()

    procs = []
    for i in range(gpus):
        reader_p = multiprocessing.Process(target=main, args=(args, i, resqueue))
        reader_p.daemon = True
        reader_p.start()
        procs.append(reader_p)

    workersprocs =[]
    _, _, i2c = get_vocab_from_file(args.i + "/vocab.txt")
    for i in range(nworkers):
        reader_p = multiprocessing.Process(target=poolProc, args=(resqueue, outqueue, i2c))
        reader_p.daemon = True
        reader_p.start()
        workersprocs.append(reader_p)


    writer = multiprocessing.Process(target=writer, args=(outqueue, args.n))
    writer.daemon=True
    writer.start()
    writer.join()
    exit()