from FPSim2.io import create_db_file
from FPSim2 import FPSim2Engine
import timeit
from tqdm import tqdm
fp_filename = 'chembl.h5'
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
with open("kinasesmiles/john_smiles_kinasei.smi", 'r') as f:
    smiles = map(lambda x : x.split(' ')[0], f.readlines())

fps = []
for smi in tqdm(smiles):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    fp1 = AllChem.GetMorganFingerprint(m, 2)
    fps.append(fp1)


with open("test.txt", 'r') as f:
    with open("out.txt", 'r') as fout:
        for smi in tqdm(f):
            smi = smi.strip()
            m = Chem.MolFromSmiles(smi)
            if m is None:
                fout.write(smi + ','+ 'NaN\n')
                continue

            fp2 = AllChem.GetMorganFingerprint(m,2)
            max_sim = 0
            for fp1 in fps:
                max_sim = max(max_sim, DataStructs.TanimotoSimilarity(fp1,fp2))
            fout.write(smi + ','+ str(max_sim) + '\n')
