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
import sys
import os
import socket

try:
    from SocketServer import ThreadingMixIn
except ImportError:
    from socketserver import ThreadingMixIn
from threading import Thread
from threading import Lock
from threading import Condition
from threading import Event

from openeye import oechem
from openeye import oeshape

try:
    from openeye import oefastrocs
except ImportError:
    oechem.OEThrow.Fatal("This script is not available, "
                         "FastROCS is not supported on this platform.")

try:
    from xmlrpclib import Binary
    from SimpleXMLRPCServer import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
except ImportError:  # python 3
    from xmlrpc.client import Binary
    from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

oepy = os.path.join(os.path.dirname(__file__), "..", "python")
sys.path.insert(0, os.path.realpath(oepy))

# very important that OEChem is in this mode since we are passing molecules between threads
oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System)


class ReadWriteLock(object):
    """ Basic locking primitive that allows multiple readers but only
    a single writer at a time. Useful for synchronizing database
    updates. Priority is given to pending writers. """
    def __init__(self):
        self.cond = Condition()
        self.readers = 0
        self.writers = 0

    def AcquireReadLock(self):
        self.cond.acquire()
        try:
            while self.writers:
                self.cond.wait()

            self.readers += 1
            assert self.writers == 0
        finally:
            self.cond.notifyAll()
            self.cond.release()

    def ReleaseReadLock(self):
        self.cond.acquire()
        assert self.readers > 0
        try:
            self.readers -= 1
        finally:
            self.cond.notifyAll()
            self.cond.release()

    def AcquireWriteLock(self):
        self.cond.acquire()
        self.writers += 1
        while self.readers:
            self.cond.wait()

        assert self.readers == 0
        assert self.writers > 0

    def ReleaseWriteLock(self):
        assert self.readers == 0
        assert self.writers > 0

        self.writers -= 1
        self.cond.notifyAll()
        self.cond.release()


class ShapeQueryThread(Thread):
    """ A thread to run a query against a shape database """

    def __init__(self, shapedb, querymolstr, nhits, iformat, oformat, errorLevel, **kwargs):
        """ Create a new thread to perform a query. The query doesn't
        execute until start is called.
        shapedb - database to run the query against

        See MCMolShapeDatabase.GetBestOverlays for a description of
        the querymolstr and nhits arguments.
        """
        Thread.__init__(self)

        self.shapeOnly = kwargs.pop('shapeOnly', False)
        self.tversky = kwargs.pop('tversky', False)
        self.altStarts = kwargs.pop('altStarts', False)
        self.randStarts = kwargs.pop('randStarts', False)

        self.shapedb = shapedb
        self.querymolstr = querymolstr
        self.iformat = iformat
        self.oformat = oformat
        self.scoretype = GetDatabaseType(self.shapeOnly)
        self.simFuncType = GetSimFuncType(self.tversky)

        numHistBins = 200
        if self.shapeOnly:
            numHistBins = 100
        self.tracer = oefastrocs.OEDBTracer(numHistBins)
        self.options = oefastrocs.OEShapeDatabaseOptions()
        self.options.SetTracer(self.tracer)
        self.options.SetLimit(nhits)
        self.options.SetScoreType(self.scoretype)
        self.options.SetSimFunc(self.simFuncType)

        if self.altStarts:
            self.options.SetInitialOrientation(GetStartType(self.altStarts))
            if self.randStarts:
                self.options.SetNumRandomStarts(self.randStarts)

        self.lock = Lock()
        self.errorLevel = errorLevel

    def run(self):
        """ Perform the query """
        # make sure the error level is set for this operating system thread
        oechem.OEThrow.SetLevel(self.errorLevel)
        try:
            results = self.shapedb.GetBestOverlays(self.querymolstr,
                                                   self.options,
                                                   self.iformat,
                                                   self.oformat)

            # since we are writing to the thread's dictionary this could
            # race with the GetStatus method below
            self.lock.acquire()
            try:
                self.results = results
                if not results:
                    self.exception = RuntimeError("Query error, no results to return, "
                                                  "check the server log for more information")
            finally:
                self.lock.release()

        except Exception as e:
            self.lock.acquire()
            try:
                self.exception = e
            finally:
                self.lock.release()

    def GetStatus(self, blocking):
        """ Returns a tuple of (count, total). count is the number of
        conformers already searched. total is the total number of
        conformers that will be searched.

        If blocking is True this method will not return until the
        count has been changed (beware of deadlocks!). If blocking is
        False the function will return immediately.
        """
        self.lock.acquire()
        try:
            if hasattr(self, "exception"):
                raise self.exception

            return self.tracer.GetCounts(blocking), self.tracer.GetTotal()
        finally:
            self.lock.release()

    def GetHistogram(self):
        """ Returns a list of integers representing the histogram of
        the molecule scores already scored.
        """
        self.lock.acquire()
        try:
            if hasattr(self, "exception"):
                raise self.exception

            hist = self.tracer.GetHistogram()
            scoretype = self.scoretype
        finally:
            self.lock.release()

        frequencies = oechem.OEUIntVector()
        hist.GetHistogram(frequencies, scoretype)
        return list(frequencies)

    def GetResults(self):
        """ Return an OEB string containing the overlaid
        confomers. This method should only be called after this thread
        has been joined. """

        if hasattr(self, "exception"):
            raise self.exception

        return self.results


class ShapeQueryThreadPool:
    """
    Maintains a pool of threads querying the same MCMolShapeDatabase.
    """
    def __init__(self, dbase):
        """ Create a new thread pool to issues queries to dbase """
        self.shapedb = dbase
        self.queryidx = 0
        self.threads = {}
        self.lock = Lock()
        self.errorLevel = oechem.OEThrow.GetLevel()

    def SubmitQuery(self, querymolstr, nhits, iformat, oformat, kwargs):
        """ Returns an index that can be passed to the QueryStatus and
        QueryResults methods.

        See MCMolShapeDatabase.GetBestOverlays for a description of
        the querymolstr and nhits arguments.
        """
        self.lock.acquire()
        try:
            idx = self.queryidx
            self.queryidx += 1
            self.threads[idx] = ShapeQueryThread(self.shapedb,
                                                 querymolstr,
                                                 nhits,
                                                 iformat,
                                                 oformat,
                                                 self.errorLevel,
                                                 **kwargs)
            self.threads[idx].start()
        finally:
            self.lock.release()

        return idx

    def QueryStatus(self, idx, blocking):
        """ Returns the status of the query indicated by idx. See
        ShapeQueryThread.GetStatus for the description of the blocking
        argument. """
        self.lock.acquire()
        try:
            thrd = self.threads[idx]
        finally:
            self.lock.release()

        return thrd.GetStatus(blocking)

    def QueryHistogram(self, idx):
        """ Returns the histogram of molecule scores already scored
        for the query indicated by idx. """
        self.lock.acquire()
        try:
            thrd = self.threads[idx]
        finally:
            self.lock.release()

        return thrd.GetHistogram()

    def QueryResults(self, idx):
        """ Wait for the query associated with idx to complete and
        then return the results as an OEB string. """
        self.lock.acquire()
        try:
            thrd = self.threads[idx]
            del self.threads[idx]
        finally:
            self.lock.release()

        thrd.join()
        return thrd.GetResults()

    def SetLevel(self, level):
        """ Set what level of information should be printed by the server. """
        self.errorLevel = level
        return True


class DatabaseLoaderThread(Thread):
    """ A thread to read a database into memory. Special note, OEChem
    must be placed in system allocation mode using
    oechem.OESetMemPoolMode(oechem.OEMemPoolMode_System). This is because the
    default OEChem memory caching scheme uses thread local storage,
    but since this thread is temporary only for reading in molecules
    that memory will be deallocated when this thread is terminated."""
    def __init__(self, shapedb, moldb, dbname, loadedEvent):
        """
        shapedb - the shapedb to add the molecules to
        moldb   - the OEMolDatabase object to use
        dbname  - the file name to open the OEMolDatabase on
        loadedEvent - event to set once loading is finished
        """
        Thread.__init__(self)
        self.shapedb = shapedb
        self.moldb = moldb
        self.dbname = dbname
        self.loadedEvent = loadedEvent

    def run(self):
        """ Open the database file and load it into the OEShapeDatabase """
        timer = oechem.OEWallTimer()
        sys.stderr.write("Opening database file %s ...\n" % self.dbname)
        if not self.moldb.Open(self.dbname):
            oechem.OEThrow.Fatal("Unable to open '%s'" % self.dbname)

        dots = oechem.OEThreadedDots(10000, 200, "conformers")
        if not self.shapedb.Open(self.moldb, dots):
            oechem.OEThrow.Fatal("Unable to initialize OEShapeDatabase on '%s'" % self.dbname)

        dots.Total()
        sys.stderr.write("%s seconds to load database\n" % timer.Elapsed())
        self.loadedEvent.set()


def SetupStream(strm, format):
    format = format.strip('.')
    ftype = oechem.OEGetFileType(format)
    if ftype == oechem.OEFormat_UNDEFINED:
        raise ValueError("Unsupported file format sent to server '%s'" % format)
    strm.SetFormat(ftype)
    strm.Setgz(oechem.OEIsGZip(format))
    return strm


OECOLOR_FORCEFIELDS = {
    "ImplicitMillsDean": oeshape.OEColorFFType_ImplicitMillsDean,
    "ImplicitMillsDeanNoRings": oeshape.OEColorFFType_ImplicitMillsDeanNoRings,
    "ExplicitMillsDean": oeshape.OEColorFFType_ExplicitMillsDean,
    "ExplicitMillsDeanNoRings": oeshape.OEColorFFType_ExplicitMillsDeanNoRings
    }


def GetDatabaseType(shapeOnly):
    if shapeOnly:
        return oefastrocs.OEShapeDatabaseType_Shape
    return oefastrocs.OEShapeDatabaseType_Default


def GetSimFuncType(simFuncType):
    if simFuncType:
        return oefastrocs.OEShapeSimFuncType_Tversky
    return oefastrocs.OEShapeSimFuncType_Tanimoto


def GetStartType(altStarts):
    if altStarts == 'random':
        return oefastrocs.OEFastROCSOrientation_Random
    if altStarts == 'inertialAtHeavyAtoms':
        return oefastrocs.OEFastROCSOrientation_InertialAtHeavyAtoms
    if altStarts == 'inertialAtColorAtoms':
        return oefastrocs.OEFastROCSOrientation_InertialAtColorAtoms
    if altStarts == 'subrocs':
        return oefastrocs.OEFastROCSOrientation_Subrocs
    return oefastrocs.OEFastROCSOrientation_Inertial


def GetAltStartsString(altStarts):
    if altStarts == oefastrocs.OEFastROCSOrientation_Random:
        return 'random'
    if altStarts == oefastrocs.OEFastROCSOrientation_InertialAtHeavyAtoms:
        return 'inertialAtHeavyAtoms'
    if altStarts == oefastrocs.OEFastROCSOrientation_InertialAtColorAtoms:
        return 'inertialAtColorAtoms'
    if altStarts == oefastrocs.OEFastROCSOrientation_Subrocs:
        return 'subrocs'
    return 'inertial'


def GetShapeDatabaseArgs(itf):
    shapeOnly = itf.GetBool("-shapeOnly")
    if shapeOnly and itf.GetParameter("-chemff").GetHasValue():
        oechem.OEThrow.Fatal("Unable to specify -shapeOnly and -chemff at the same time!")

    chemff = itf.GetString("-chemff")
    if not chemff.endswith(".cff"):
        return (GetDatabaseType(shapeOnly), OECOLOR_FORCEFIELDS[chemff])

    # given a .cff file, use that to construct a OEColorForceField
    assert not shapeOnly
    cff = oeshape.OEColorForceField()
    if not cff.Init(chemff):
        oechem.OEThrow.Fatal("Unable to read color force field from '%s'" % chemff)

    return (cff,)


def ReadShapeQuery(querymolstr):
    iss = oechem.oeisstream(querymolstr)
    query = oeshape.OEShapeQueryPublic()

    if not oeshape.OEReadShapeQuery(iss, query):
        raise ValueError("Unable to read a shape query from the data string")

    return query


class MCMolShapeDatabase:
    """ Maintains a database of MCMols that can be queried by shape
    similarity."""
    def __init__(self, itf):
        """ Create a MCMolShapeDatabase from the parameters specified by the OEInterface. """
        self.rwlock = ReadWriteLock()
        self.loadedEvent = Event()

        self.dbname = itf.GetString("-dbase")
        self.moldb = oechem.OEMolDatabase()

        self.dbtype = GetDatabaseType(itf.GetBool("-shapeOnly"))
        self.shapedb = oefastrocs.OEShapeDatabase(*GetShapeDatabaseArgs(itf))

        # this thread is daemonic so a KeyboardInterupt
        # during the load will cancel the process
        self.loaderThread = DatabaseLoaderThread(self.shapedb,
                                                 self.moldb,
                                                 self.dbname,
                                                 self.loadedEvent)
        self.loaderThread.setDaemon(True)
        self.loaderThread.start()

    def IsLoaded(self, blocking=False):
        """ Return whether the server has finished loading. """
        if blocking:
            self.loadedEvent.wait()

        # clean up the load waiter thread if it's still there
        if self.loadedEvent.isSet() and self.loaderThread is not None:
            self.rwlock.AcquireWriteLock()
            try:  # typical double checked locking
                if self.loaderThread is not None:
                    self.loaderThread.join()
                    self.loaderThread = None
            finally:
                self.rwlock.ReleaseWriteLock()

        return self.loadedEvent.isSet()

    def GetBestOverlays(self, querymolstr, options, iformat, oformat):
        """ Return a string of the format specified by 'oformat'
        containing nhits overlaid confomers using querymolstr as the
        query interpretted as iformat.

        querymolstr - a string containing a molecule to use as the query
        options - an instance of OEShapeDatabaseOptions
        iformat - a string representing the file extension to parse the querymolstr as.
                  Note: old clients could be passing .sq files, so
                  iformat == '.oeb' will try to interpret the file as
                  a .sq file.
        oformat - file format to write the results as
        """
        timer = oechem.OEWallTimer()

        # make sure to wait for the load to finish
        blocking = True
        loaded = self.IsLoaded(blocking)
        assert loaded

        if iformat.startswith(".sq"):
            query = ReadShapeQuery(querymolstr)
        else:
            # read in query
            qfs = oechem.oemolistream()
            qfs = SetupStream(qfs, iformat)
            if not qfs.openstring(querymolstr):
                raise ValueError("Unable to open input molecule string")

            query = oechem.OEGraphMol()
            if not oechem.OEReadMolecule(qfs, query):
                if iformat == ".oeb":  # could be an old client trying to send a .sq file.
                    query = ReadShapeQuery(querymolstr)
                else:
                    raise ValueError("Unable to read a molecule from the string of format '%s'"
                                     % iformat)

        ofs = oechem.oemolostream()
        ofs = SetupStream(ofs, oformat)
        if not ofs.openstring():
            raise ValueError("Unable to openstring for output")

        # do we only want shape based results?

        # this is a "Write" lock to be paranoid and not overload the GPU
        self.rwlock.AcquireWriteLock()
        try:
            # do search
            scores = self.shapedb.GetSortedScores(query, options)
            sys.stderr.write("%f seconds to do search\n" % timer.Elapsed())
        finally:
            self.rwlock.ReleaseWriteLock()

        timer.Start()
        # write results
        for score in scores:
            mcmol = oechem.OEMol()
            if not self.moldb.GetMolecule(mcmol, score.GetMolIdx()):
                oechem.OEThrow.Warning("Can't retrieve molecule %i from the OEMolDatabase, "
                                       "skipping..." % score.GetMolIdx())
                continue
            # remove hydrogens to make output smaller, this also
            # ensures OEPrepareFastROCSMol will have the same output
            oechem.OESuppressHydrogens(mcmol)

            mol = oechem.OEGraphMol(mcmol.GetConf(oechem.OEHasConfIdx(score.GetConfIdx())))
            oechem.OECopySDData(mol, mcmol)

            if options.GetSimFunc() == oefastrocs.OEShapeSimFuncType_Tanimoto:
                oechem.OESetSDData(mol, "ShapeTanimoto", "%.4f" % score.GetShapeTanimoto())
                oechem.OESetSDData(mol, "ColorTanimoto", "%.4f" % score.GetColorTanimoto())
                oechem.OESetSDData(mol, "TanimotoCombo", "%.4f" % score.GetTanimotoCombo())
            else:
                oechem.OESetSDData(mol, "ShapeTversky", "%.4f" % score.GetShapeTversky())
                oechem.OESetSDData(mol, "ColorTversky", "%.4f" % score.GetColorTversky())
                oechem.OESetSDData(mol, "TverskyCombo", "%.4f" % score.GetTverskyCombo())

            if options.GetInitialOrientation() != oefastrocs.OEFastROCSOrientation_Inertial:
                oechem.OEAddSDData(mol, "Opt. Starting Pos.",
                                   GetAltStartsString(options.GetInitialOrientation()))

            score.Transform(mol)

            oechem.OEWriteMolecule(ofs, mol)

        output = ofs.GetString()
        sys.stderr.write("%f seconds to write hitlist\n" % timer.Elapsed())
        sys.stderr.flush()
        ofs.close()

        return output

    def GetName(self):
        self.rwlock.AcquireReadLock()
        try:
            return self.dbname
        finally:
            self.rwlock.ReleaseReadLock()

    def SetName(self, name):
        self.rwlock.AcquireWriteLock()
        try:
            self.dbname = name
        finally:
            self.rwlock.ReleaseWriteLock()


class ShapeQueryServer:
    """ This object's methods are exposed via XMLRPC. """
    def __init__(self, itf):
        """ Initialize the server to serve queries on the database
        named by dbname."""
        self.shapedb = MCMolShapeDatabase(itf)
        self.thdpool = ShapeQueryThreadPool(self.shapedb)
        self.itf = itf

    def IsLoaded(self, blocking=False):
        """ Return whether the server has finished loading. """
        return self.shapedb.IsLoaded(blocking)

    def GetBestOverlays(self, querymolstr, nhits, iformat=".oeb", oformat=".oeb"):
        """ A blocking call that only returns once the query is completed. """
        results = self.shapedb.GetBestOverlays(querymolstr.data, nhits, iformat, oformat)
        return Binary(results)

    def SubmitQuery(self, querymolstr, nhits, iformat=".oeb", oformat=".oeb", kwargs=None):
        """ Returns a index that can be used by QueryStatus and
        QueryResults. This method will return immediately."""
        if not kwargs:
            kwargs = {}
        if self.itf.GetBool("-shapeOnly"):
            kwargs['shapeOnly'] = True

        return self.thdpool.SubmitQuery(querymolstr.data, nhits, iformat, oformat, kwargs)

    def QueryStatus(self, queryidx, blocking=False):
        """ Return the status of the query specified by queryidx. See
        ShapeQueryThread.GetStatus for a description of the blocking
        argument and the return value."""
        return self.thdpool.QueryStatus(queryidx, blocking)

    def QueryHistogram(self, queryidx):
        """ Return the current histogram of scores specified by
        queryidx."""
        return self.thdpool.QueryHistogram(queryidx)

    def QueryResults(self, queryidx):
        """ Wait for the query associated with idx to complete and
        then return the results as an OEB string. """
        results = self.thdpool.QueryResults(queryidx)
        return Binary(results)

    def GetVersion(self):
        """ Returns what version of FastROCS this server is. """
        return oefastrocs.OEFastROCSGetRelease()

    def OEThrowSetLevel(self, level):
        """ Set what level of information should be printed by the server. """
        return self.thdpool.SetLevel(level)

    def GetName(self):
        """ The name of this database. By default this is the file name of the database used. """
        return self.shapedb.GetName()

    def SetName(self, name):
        """ Set a custom database name for this server. """
        self.shapedb.SetName(name)
        return True


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


class AsyncXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    # if a shutdown request occurs through a signal force everything to terminate immediately
    daemon_threads = True
    allow_reuse_address = True


InterfaceData = """\
!BRIEF [-shapeOnly | -chemff <color forcefield>] [-hostname] [-dbase] database [[-port] 8080]
!PARAMETER -dbase
  !TYPE string
  !REQUIRED true
  !BRIEF Input database to serve
  !KEYLESS 1
!END
!PARAMETER -port
  !TYPE int
  !REQUIRED false
  !BRIEF Port number to start the XML RPC server on
  !DEFAULT 8080
  !KEYLESS 2
!END
!PARAMETER -hostname
  !TYPE string
  !DEFAULT 0.0.0.0
  !BRIEF Name of the server to bind to
!END
!PARAMETER -shapeOnly
  !ALIAS -s
  !TYPE bool
  !DEFAULT false
  !BRIEF Run FastROCS server in shape only mode, clients can also control this separately
!END
!PARAMETER -chemff
  !TYPE string
  !LEGAL_VALUE ImplicitMillsDean
  !LEGAL_VALUE ImplicitMillsDeanNoRings
  !LEGAL_VALUE ExplicitMillsDean
  !LEGAL_VALUE ExplicitMillsDeanNoRings
  !LEGAL_VALUE *.cff
  !DEFAULT ImplicitMillsDean
  !BRIEF Chemical force field. Either a constant or a filename.
!END
"""


def main(argv=[__name__]):

    if not oefastrocs.OEFastROCSIsGPUReady():
        oechem.OEThrow.Fatal("No supported GPU available to run FastROCS TK!")

    itf = oechem.OEInterface(InterfaceData, argv)

    # default hostname to bind is 0.0.0.0, to allow connections with
    # any hostname
    hostname = itf.GetString("-hostname")

    # default port number is 8080
    portnumber = itf.GetInt("-port")

    # create server
    server = AsyncXMLRPCServer((hostname, portnumber),
                               requestHandler=RequestHandler,
                               logRequests=False)
    hostname, portnumber = server.socket.getsockname()
    if hostname == "0.0.0.0":
        hostname = socket.gethostname()
    sys.stderr.write("Listening for ShapeDatabaseClient.py requests on %s:%i\n\n"
                     % (hostname, portnumber))
    sys.stderr.write("Example: ShapeDatabaseClient.py %s:%i query.sdf hit.sdf\n\n"
                     % (hostname, portnumber))

    # register the XMLRPC methods
    server.register_introspection_functions()

    server.register_instance(ShapeQueryServer(itf))

    try:
        # Run the server's main loop
        server.serve_forever()
    finally:
        server.server_close()

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))