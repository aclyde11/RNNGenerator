import sys
from openeye import oechem
from openeye import oeshape
from openeye import oeomega
from tqdm import tqdm
from glob import glob
import numpy as np
import multiprocessing

def run_one_roc(mol):
    fitfs = oechem.oemolistream("test.sdf")

    options = oeshape.OEROCSOptions()
    options.SetNumBestHits(1)
    rocs = oeshape.OEROCS(options)

    rocs.SetDatabase(fitfs)
    max_score = 0
    for res in rocs.Overlay(mol):
        max_score = max(res.GetTanimotoCombo(), max_score)
    return max_score

def test_confomrmers(mols):
    pool = multiprocessing.Pool(10)
    res = pool.imap_unordered(run_one_roc, mols)
    max_score = 0
    for i in tqdm(res, total=len(mols)):
        max_score = max(max_score, i)
    print(max_score)

def FromMol(mol, isomer=True, num_enantiomers=-1):
    """
    Generates a set of conformers as an OEMol object
    Inputs:
        mol is an OEMol
        isomers is a boolean controling whether or not the various diasteriomers of a molecule are created
        num_enantiomers is the allowable number of enantiomers. For all, set to -1
    """
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetMaxConfs(199)
    omega = oeomega.OEOmega(omegaOpts)
    out_conf = []
    ofs = oechem.oemolostream("test.sdf")
    if not isomer:
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            out_conf.append(mol)
        else:
            oechem.OEThrow.Warning("%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code)))

    elif isomer:
        for enantiomer in oeomega.OEFlipper(mol.GetActive(), 12, True):
            enantiomer = oechem.OEMol(enantiomer)
            ret_code = omega.Build(enantiomer)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                out_conf.append(enantiomer)
                num_enantiomers -= 1
                oechem.OEWriteMolecule(ofs, mol)
                if num_enantiomers == 0:
                    break
            else:
                oechem.OEThrow.Warning("%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code)))

    return out_conf


def FromString(smiles, isomer=True, num_enantiomers=1):
    """
    Generates an set of conformers from a SMILES string
    """
    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles):
        print("SMILES invalid for string", smiles)
        return None
    else:
        return FromMol(mol, isomer, num_enantiomers)


def rocs(refmol):
    fitfs = oechem.oemolistream("test.sdf")


    options = oeshape.OEROCSOptions()
    options.SetNumBestHits(1)
    rocs = oeshape.OEROCS(options)

    rocs.SetDatabase(fitfs)

    mols = []
    for f in tqdm(glob('/Users/austin/Downloads/template_ligands/*.mol2')):
        ifs = oechem.oemolistream(f)
        mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, mol)
        mols.append(mol)

    test_confomrmers(mols)
    # max_score = 0
    # for m in tqdm(mols):
    #     for res in rocs.Overlay(m):
    #         # outmol = res.GetOverlayConfs()
    #         max_score = max(res.GetTanimotoCombo(), max_score)



def main(argv):

    # s = FromString("CC(=O)OC1=CC=CC=C1C(=O)O")[0]
    s = FromString("C1CCC(C1)C(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3")[0]

    confs = FromMol(s)
    print(len(confs))
    print(confs[0].GetMaxConfIdx())

    rocs(None)



main(None)

