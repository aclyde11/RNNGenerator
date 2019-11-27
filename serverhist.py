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
from __future__ import unicode_literals
import os
import sys
import argparse

try:
    from xmlrpclib import ServerProxy, Binary, Fault
except ImportError:  # python 3
    from xmlrpc.client import ServerProxy, Binary, Fault


class Pyasciigraph:
    """ Copied from https://pypi.python.org/pypi/ascii_graph/0.2.1 """
    def __init__(self, line_length=79, min_graph_length=50, separator_length=2):
        """Constructor of Pyasciigraph

        :param int line_length: the max number of char on a line
                if any line cannot be shorter,
                it will go over this limit
        :param int min_graph_length: the min number of char used by the graph
        :param int separator_length: the length of field separator
        """
        self.line_length = line_length
        self.separator_length = separator_length
        self.min_graph_length = min_graph_length

    def _u(self, x):
        if sys.version < '3':
            import codecs
            return codecs.unicode_escape_decode(x)[0]
        else:
            return x

    def _get_maximum(self, data):
        all_max = {}
        all_max['value_max_length'] = 0
        all_max['info_max_length'] = 0
        all_max['max_value'] = 0

        for (info, value) in data:
            if value > all_max['max_value']:
                all_max['max_value'] = value

            if len(info) > all_max['info_max_length']:
                all_max['info_max_length'] = len(info)

            if len(str(value)) > all_max['value_max_length']:
                all_max['value_max_length'] = len(str(value))
        return all_max

    def _gen_graph_string(self, value, max_value, graph_length, start_value):
        number_of_square = 0
        if max_value:
            number_of_square = int(value * graph_length / max_value)
        number_of_space = int(start_value - number_of_square)
        return '#' * number_of_square + self._u(' ') * number_of_space

    def _gen_info_string(self, info, start_info, line_length):
        number_of_space = (line_length - start_info - len(info))
        return info + self._u(' ') * number_of_space

    def _gen_value_string(self, value, start_value, start_info):
        number_space = start_info -\
                start_value -\
                len(str(value)) -\
                self.separator_length

        return ' ' * number_space +\
            str(value) +\
            ' ' * self.separator_length

    def _sanitize_string(self, string):
        # get the type of a unicode string
        unicode_type = type(self._u('t'))
        input_type = type(string)
        if input_type is str:
            info = string
        elif input_type is unicode_type:
            info = string
        elif input_type is int or input_type is float:
            info = str(string)
        return info

    def _sanitize_data(self, data):
        ret = []
        for item in data:
            ret.append((self._sanitize_string(item[0]), item[1]))
        return ret

    def graph(self, label, data, sort=0, with_value=True):
        """function generating the graph

        :param string label: the label of the graph
        :param iterable data: the data (list of tuple (info, value))
                info must be "castable" to a unicode string
                value must be an int or a float
        :param int sort: flag sorted
                0: not sorted (same order as given) (default)
                1: increasing order
                2: decreasing order
        :param boolean with_value: flag printing value
                True: print the numeric value (default)
                False: don't print the numeric value
        :rtype: a list of strings (each lines)

        """
        result = []
        san_data = self._sanitize_data(data)
        san_label = self._sanitize_string(label)

        if sort == 1:
            san_data = sorted(san_data, key=lambda value: value[1], reverse=False)
        elif sort == 2:
            san_data = sorted(san_data, key=lambda value: value[1], reverse=True)

        all_max = self._get_maximum(san_data)

        real_line_length = max(self.line_length, len(label))

        min_line_length = self.min_graph_length + 2 * self.separator_length +\
            all_max['value_max_length'] + all_max['info_max_length']

        if min_line_length < real_line_length:
            # calcul of where to start info
            start_info = self.line_length -\
                all_max['info_max_length']
            # calcul of where to start value
            start_value = start_info -\
                self.separator_length -\
                all_max['value_max_length']
            # calcul of where to end graph
            graph_length = start_value -\
                self.separator_length
        else:
            # calcul of where to start value
            start_value = self.min_graph_length +\
                self.separator_length
            # calcul of where to start info
            start_info = start_value +\
                all_max['value_max_length'] +\
                self.separator_length
            # calcul of where to end graph
            graph_length = self.min_graph_length
            # calcul of the real line length
            real_line_length = min_line_length

        result.append(san_label)
        result.append(self._u('#') * real_line_length)

        for item in san_data:
            info = item[0]
            value = item[1]

            graph_string = self._gen_graph_string(
                value,
                all_max['max_value'],
                graph_length,
                start_value
                )

            value_string = self._gen_value_string(
                value,
                start_value,
                start_info
                )

            info_string = self._gen_info_string(
                info,
                start_info,
                real_line_length
                )
            new_line = graph_string + value_string + info_string
            result.append(new_line)

        return result


def AddBin(bins, binSize, binIdx, curTotal):
    lowerBound = binSize * binIdx
    label = "%.2f" % lowerBound
    bins.append((label, curTotal))


def GetGraphTitle(tversky, shapeOnly):
    if not tversky and not shapeOnly:
        return "FastROCS Tanimoto Combo Score Distribution"
    if not tversky and shapeOnly:
        return "FastROCS Tanimoto Shape Score Distribution"
    if tversky and not shapeOnly:
        return "FastROCS Tversky Combo Score Distribution"
    if tversky and shapeOnly:
        return "FastROCS Tversky Shape Score Distribution"


def PrintHistogram(hist, tversky=None, shapeOnly=None):
    squashFactor = 10
    if shapeOnly:
        maxScore = 1.0
    else:
        maxScore = 2.0
    binSize = maxScore/(len(hist) / squashFactor)

    bins = []
    curTotal = 0
    binIdx = 0
    for i, val in enumerate(hist):
        if i != 0 and (i % squashFactor) == 0:
            AddBin(bins, binSize, binIdx, curTotal)
            curTotal = 0
            binIdx += 1

        curTotal += val
    AddBin(bins, binSize, binIdx, curTotal)

    graph = Pyasciigraph()

    for line in graph.graph(GetGraphTitle(tversky, shapeOnly), bins):
        print(line)


def GetFormatExtension(fname):
    base, ext = os.path.splitext(fname.lower())
    if ext == ".gz":
        base, ext = os.path.splitext(base)
        ext += ".gz"
    return ext


def main(argv=[__name__]):

    parser = argparse.ArgumentParser()

    # positional arguments retaining backward compatibility
    parser.add_argument('server:port', help='Server name and port number \
                        of database to search i.e. localhost:8080.')
    parser.add_argument('query', help='File containing the query molecule to search \
                        (format not restricted to *.oeb).')
    parser.add_argument('results', help='Output file to store results \
                        (format not restricted to *.oeb).')
    parser.add_argument('nHits', nargs='?', type=int, default=100,
                        help='Number of hits to return (default=100).')
    parser.add_argument('--tversky', action='store_true', default=argparse.SUPPRESS,
                        help='Switch to Tversky similarity scoring (default=Tanimoto).')
    parser.add_argument('--shapeOnly', action='store_true', default=argparse.SUPPRESS,
                        help='Switch to shape-only scores (default=Combo).')
    parser.add_argument('--alternativeStarts', default=argparse.SUPPRESS, nargs=1, dest='altStarts',
                        choices=('random', 'subrocs',
                                 'inertialAtHeavyAtoms', 'inertialAtColorAtoms'),
                        help='Optimize using alternative starts. '
                             'To perform N random starts do \
                             "--alternativeStarts random N" (default N=10)')

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
        sys.stderr.write(str(e))
        return 1

    while True:
        blocking = True
        try:
            current, total = s.QueryStatus(idx, blocking)
            hist = s.QueryHistogram(idx)
        except Fault as e:
            print(str(e), file=sys.stderr)
            return 1

        if total == 0:
            continue

        PrintHistogram(hist, dargs.get('tversky', None), dargs.get('shapeOnly', None))

        if total <= current:
            break

    results = s.QueryResults(idx)

    # assuming the results come back as a string in the requested format
    with open(ofname, 'wb') as output:
        output.write(results.data)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))