#!/usr/bin/env python
# (C) 2017 OpenEye Scientific Software Inc. All rights reserved.
#
# TERMS FOR USE OF SAMPLE CODE The software below ("Sample Code") is
# provided to current licensees or subscribers of OpenEye products or
# SaaS offerings (each a "Customer").
# Customer is hereby permitted to use, copy, and modify the Sample Code,
# subject to these terms. OpenEye claims no rights to Customer's
# modifications. Modification of Sample Code is at Customer's sole and
# exclusive risk. Sample Code may require Customer to have a then
# current license or subscription to the applicable OpenEye offering.
# THE SAMPLE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED.  OPENEYE DISCLAIMS ALL WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. In no event shall OpenEye be
# liable for any damages or liability in connection with the Sample Code
# or its use.

from __future__ import print_function
import os
import sys
import argparse

from openeye import oechem
from openeye import oefastrocs

oepy = os.path.join(os.path.dirname(__file__), "..", "python")
sys.path.insert(0, os.path.realpath(oepy))


def main(argv=[__name__]):

    parser = argparse.ArgumentParser()

    # positional arguments retaining backward compatibility
    parser.add_argument('database',
                        help='File containing the database molecules to be search \
                              (format not restricted to *.oeb).')
    parser.add_argument('query', default=[], nargs='+',
                        help='File containing the query molecule(s) to be search \
                              (format not restricted to *.oeb).')
    parser.add_argument('--nHits', dest='nHits', type=int, default=100,
                        help='Number of hits to return (default = number of database mols).')
    parser.add_argument('--cutoff',  dest='cutoff', type=float, default=argparse.SUPPRESS,
                        help='Specify a cutoff criteria for scores.')
    parser.add_argument('--tversky', dest='tversky', action='store_true', default=argparse.SUPPRESS,
                        help='Switch to Tversky similarity scoring (default = Tanimoto).')

    args = parser.parse_args()

    dbname = args.database

    if not oefastrocs.OEFastROCSIsGPUReady():
        oechem.OEThrow.Info("No supported GPU available!")
        return 0

    # set options
    opts = oefastrocs.OEShapeDatabaseOptions()
    opts.SetLimit(args.nHits)
    print("Number of hits set to %u" % opts.GetLimit())
    if hasattr(args, 'cutoff') is not False:
        opts.SetCutoff(args.cutoff)
        print("Cutoff set to %f" % args.cutoff)
    if hasattr(args, 'tversky') is not False:
        opts.SetSimFunc(args.tversky)
        print("Tversky similarity scoring set.")

    # read in database
    ifs = oechem.oemolistream()
    if not ifs.open(dbname):
        oechem.OEThrow.Fatal("Unable to open '%s'" % dbname)

    print("\nOpening database file %s ..." % dbname)
    timer = oechem.OEWallTimer()
    dbase = oefastrocs.OEShapeDatabase()
    moldb = oechem.OEMolDatabase()
    if not moldb.Open(ifs):
        oechem.OEThrow.Fatal("Unable to open '%s'" % dbname)

    dots = oechem.OEThreadedDots(10000, 200, "conformers")
    if not dbase.Open(moldb, dots):
        oechem.OEThrow.Fatal("Unable to initialize OEShapeDatabase on '%s'" % dbname)

    dots.Total()
    print("%f seconds to load database\n" % timer.Elapsed())

    for qfname in args.query:

        # read in query
        qfs = oechem.oemolistream()
        if not qfs.open(qfname):
            oechem.OEThrow.Fatal("Unable to open '%s'" % qfname)

        mcmol = oechem.OEMol()
        if not oechem.OEReadMolecule(qfs, mcmol):
            oechem.OEThrow.Fatal("Unable to read query from '%s'" % qfname)
        qfs.rewind()

        ext = oechem.OEGetFileExtension(qfname)

        qmolidx = 0
        while oechem.OEReadMolecule(qfs, mcmol):

            # write out to file name based on molecule title
            ofs = oechem.oemolostream()
            moltitle = mcmol.GetTitle()
            if len(moltitle) == 0:
                moltitle = str(qmolidx)
            ofname = moltitle + "_results." + ext
            if not ofs.open(ofname):
                oechem.OEThrow.Fatal("Unable to open '%s'" % argv[4])

            print("Searching for %s of %s (%s conformers)" % (moltitle, qfname, mcmol.NumConfs()))

            qconfidx = 0
            for conf in mcmol.GetConfs():

                for score in dbase.GetSortedScores(conf, opts):

                    dbmol = oechem.OEMol()
                    dbmolidx = score.GetMolIdx()
                    if not moldb.GetMolecule(dbmol, dbmolidx):
                        print("Unable to retrieve molecule '%u' from the database" % dbmolidx)
                        continue

                    mol = oechem.OEGraphMol(dbmol.GetConf(oechem.OEHasConfIdx(score.GetConfIdx())))

                    oechem.OESetSDData(mol, "QueryConfidx", "%s" % qconfidx)
                    oechem.OESetSDData(mol, "ShapeTanimoto", "%.4f" % score.GetShapeTanimoto())
                    oechem.OESetSDData(mol, "ColorTanimoto", "%.4f" % score.GetColorTanimoto())
                    oechem.OESetSDData(mol, "TanimotoCombo", "%.4f" % score.GetTanimotoCombo())
                    score.Transform(mol)

                    oechem.OEWriteMolecule(ofs, mol)

                qconfidx += 1

            print("%s conformers processed" % qconfidx)
            print("Wrote results to %s\n" % ofname)

        qmolidx += 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))