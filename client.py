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

try:
    from xmlrpclib import ServerProxy, Binary, Fault
except ImportError:  # python 3
    from xmlrpc.client import ServerProxy, Binary, Fault


def GetFormatExtension(fname):
    base, ext = os.path.splitext(fname.lower())
    if ext == ".gz":
        base, ext = os.path.splitext(base)
        ext += ".gz"
    return ext
d

def main(argv=[__name__]):

    parser = argparse.ArgumentParser()

    # positional arguments retaining backward compatibility
    parser.add_argument('server:port', help='Server name and port number of database to search '
                                            'i.e. localhost:8080.')
    parser.add_argument('query', help='File containing the query molecule to search '
                                      '(format not restricted to *.oeb).')
    parser.add_argument('results',
                        help='Output file to store results (format not restricted to *.oeb).')
    parser.add_argument('nHits',  nargs='?', type=int, default=100,
                        help='Number of hits to return (default=100).')
    parser.add_argument('--tversky', action='store_true', default=argparse.SUPPRESS,
                        help='Switch to Tversky similarity scoring (default=Tanimoto).')
    parser.add_argument('--shapeOnly', action='store_true', default=argparse.SUPPRESS,
                        help='Switch to shape-only scores (default=Combo).')
    parser.add_argument('--alternativeStarts', default=argparse.SUPPRESS, nargs=1, dest='altStarts',
                        choices=('random', 'subrocs',
                                 'inertialAtHeavyAtoms', 'inertialAtColorAtoms'),
                        help='Optimize using alternative starts (default=inertial). '
                             'To perform N random starts do '
                             '"--alternativeStarts random N" (default N=10)')

    known, remaining = (parser.parse_known_args())
    dargs = vars(known)

    qfname = dargs.pop('query')
    numHits = dargs.pop('nHits')

    startType = dargs.get('altStarts', None)

    if startType:
        dargs['altStarts'] = str(startType[0])
        if len(remaining) == 1 and dargs['altStarts'] == 'random':
            try:
                numRands = int(remaining[0])
                dargs['randStarts'] = numRands
            except ValueError:
                print("Invalid argument given. See --help menu for argument list")
                sys.exit()
        if len(remaining) > 1:
            print("Too many arguments given. See --help menu for argument list")
            sys.exit()
    else:
        if remaining:
            print("Too many arguments given. See --help menu for argument list")
            sys.exit()

    try:
        fh = open(qfname, 'rb')
    except IOError:
        sys.stderr.write("Unable to open '%s' for reading" % qfname)
        return 1

    iformat = GetFormatExtension(qfname)

    ofname = dargs.pop('results')
    oformat = GetFormatExtension(ofname)

    s = ServerProxy("http://" + dargs.pop('server:port'))
    data = Binary(fh.read())

    try:
        idx = s.SubmitQuery(data, numHits, iformat, oformat, dargs)
    except Fault as e:
        if "TypeError" in e.faultString:
            # we're trying to run against an older server, may be able
            # to still work if the formats ameniable.
            if ((iformat == ".oeb" or iformat == ".sq") and oformat == ".oeb"):
                idx = s.SubmitQuery(data, numHits)
            else:
                sys.stderr.write("%s is too new of a version to work with the server %s\n"
                                 % (argv[0], argv[1]))
                sys.stderr.write("Please upgrade your server to FastROCS version 1.4.0"
                                 " or later to be able to use this client\n")
                sys.stderr.write("This client will work with this version of the server "
                                 "if the input file is either"
                                 "'.oeb' or '.sq' and the output file is '.oeb'\n")
                return 1
        else:
            sys.stderr.write(str(e))
            return 1

    first = False
    while True:
        blocking = True
        try:
            current, total = s.QueryStatus(idx, blocking)
        except Fault as e:
            print(str(e), file=sys.stderr)
            return 1

        if total == 0:
            continue

        if first:
            print("%s/%s" % ("current", "total"))
            first = False
        print("%i/%i" % (current, total))

        if total <= current:
            break

    results = s.QueryResults(idx)

    # assuming the results come back as a string in the requested format
    with open(ofname, 'wb') as output:
        output.write(results.data)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))