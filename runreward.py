import pandas as pd
import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem.QED import qed
from sklearn.preprocessing import MinMaxScaler
from SA_Score import sascorer

def get_sa(smi):
    mol = Chem.MolFromSmiles(smi)
    return sascorer.calculateScore(mol)

def get_qed(smi):
    mol = Chem.MolFromSmiles(smi)
    return qed(mol)

def get_counts(smi):
    mol = Chem.MolFromSmiles(smi)
    return len(mol.GetAtoms())



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input csv with fastroc and sim columns already computed', type=str)
    parser.add_argument('-o', help='output csv location', type=str)
    parser.add_argument('n', help='how many to sample', type=int)
    return parser.parse_args()

if '__name__' == '__main__':
    args = get_args()
    df = pd.read_csv(args.i)
    print("Loaded csv with", df.shape[0], "rows")

    assert(('fastroc' in df.columns.tolist()) and ('sim' in df.columns.tolist()) and ('smiles' in df.columns.tolist()))

    df = df[df.sim <= 0.6]
    df['molsize'] = df.smiles.apply(get_counts)
    df['qed'] = df.smiles.apply(get_counts)
    df['sa'] = df.smiles.apply(get_sa)

    df = df.dropna()
    print("Loaded csv with", df.shape[0], "rows")

    mm = MinMaxScaler()
    df.iloc[:, 1:] = mm.fit_transform(df.iloc[:, 1:])

    w1 = -1.0 # sim
    w2 = 5.0 # fast roc
    w3 = 5.0 #size
    w4 = 1.0 # qed
    w5 = 1.0 # sa

    reward_func = lambda x : w1 * x['sim'] + w2 * x['fastroc'] + w3 * x['molsize'] + w4 * x['qed'] + w5 * x['sa']
    df['reward'] = df.apply(reward_func)

    df.iloc[:, 1:-1] = mm.inverse_transform(df.iloc[:, 1:-1])

    output = df.sort_values('reward', ascending=False).iloc[:args.n]

    output.to_csv(args.o + ".csv", index=False)
    output.smiles.to_csv(args.o + ".txt", index=False, header=False)