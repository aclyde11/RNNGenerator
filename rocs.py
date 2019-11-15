import sys
from openeye import oechem
from openeye import oeshape

def main(argv):
    if len(argv) != 5:
        oechem.OEThrow.Usage("%s <reffile> <fitfile> <outfile> <confsperhit>" % argv[0])

    reffs = oechem.oemolistream(sys.argv[1])
    fitfs = oechem.oemolistream(sys.argv[2])
    nconfs = 1

    refmol = oechem.OEMol()
    oechem.OEReadMolecule(reffs, refmol)


    options = oeshape.OEROCSOptions()
    options.SetNumBestHits(3)
    options.SetConfsPerHit(nconfs)
    rocs = oeshape.OEROCS(options)
    rocs.SetDatabase(fitfs)
    for res in rocs.Overlay(refmol):
        outmol = res.GetOverlayConfs()
        print("title: %s  tanimoto combo = %.2f" % (outmol.GetTitle(), res.GetTanimotoCombo()))

if __name__ == "__main__":
    sys.exit(main(sys.argv))