from FPSim2.io import create_db_file
from FPSim2 import FPSim2Engine
import timeit
fp_filename = 'chembl.h5'
query = ['CC(=O)Oc1ccccc1C(=O)O', 'CC(=O)Oc1ccccc1C(=O)O', 'CC(=O)Oc1ccccc1C(=O)O', 'CC(=O)Oc1ccccc1C(=O)O', 'CC(=O)Oc1ccccc1C(=O)O', 'CC(=O)Oc1ccccc1C(=O)O']

fpe = FPSim2Engine(fp_filename)

results_ = []
for q in query:
    results = fpe.similarity(q, 0.5, n_workers=4)
    results_.append(results)


x = timeit.timeit('[fpe.similarity(q, 0.5, n_workers=1) for q in query]', number=100, setup="from __main__ import fpe, query")
print(x )