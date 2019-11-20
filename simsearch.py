import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm



def compute_fp_dict(source="kinasesmiles/john_smiles_kinasei.smi"):
    with open(source, 'r') as f:
        smiles = map(lambda x: x.split(' ')[0], f.readlines())

    fps = []
    for smi in tqdm(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp1 = AllChem.GetMorganFingerprint(m, 2)
        fps.append(fp1)
    return fps

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str)
    parser.add_argument('-o', required=True, type=str)
    return parser.parse_args()

args = get_args()

fps = compute_fp_dict()
print("loaded fps")

df = pd.read_csv(args.i)
print("loaded", df.shape[0], 'smiles')
res = []
for smi in tqdm(df.loc[:, 'smiles'].tolist()):
    smi = smi.strip()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        res.append(np.nan)
        continue

    fp2 = AllChem.GetMorganFingerprint(m, 2)
    max_sim = 0
    for fp1 in fps:
        max_sim = max(max_sim, DataStructs.TanimotoSimilarity(fp1, fp2))
    res.append(max_sim)

df['sim'] = res

df.to_csv(args.o, index=False)
