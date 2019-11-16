import sys
from openeye import oechem
from openeye import oeshape
from openeye import oeomega
from tqdm import tqdm

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
    fitfs = oechem.oemolistream("/Users/austin/Downloads/template_ligands/alls.mol2")

    options = oeshape.OEROCSOptions()
    options.SetNumBestHits(1)
    rocs = oeshape.OEROCS(options)
    rocs.SetDatabase(fitfs)
    for res in tqdm(rocs.Overlay(refmol)):
        outmol = res.GetOverlayConfs()
        print("title: %s  tanimoto combo = %.2f" % (outmol.GetTitle(), res.GetTanimotoCombo(),))

def main(argv):

    s = FromString("CC(=O)OC1=CC=CC=C1C(=O)O")[0]
    s = FromString("C1CCC(C1)C(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3")[0]

    print(s)
    confs = FromMol(s)
    print(len(confs))
    print(confs[0].GetMaxConfIdx())
    for i in confs[0].GetConfs():
        print(i)
    rocs(confs[0])



main(None)

