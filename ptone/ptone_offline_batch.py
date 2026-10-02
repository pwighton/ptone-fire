#!/usr/bin/env python3
# Batch-process previously acquired pilot tone data through pilottone.py (offline config).
#
# For each subject directory in inputBaseDir matching --inputDirFilter (e.g. 'ptoneH*'):
#   - each Siemens raw data file (--measDatFilter, default 'meas_*.dat') is converted to MRD with
#     siemens_to_ismrmrd (last measurement only), sent through an MRD server running pilottone with the
#     --mrdClientConfig config (default pilottone_offline), and the temporary .h5 deleted.  Results go to
#     outputBaseDir/<subject>, named after the subject (outputFileStem)
#   - each new result is plotted with ptone/ptone_plot.py, with head motion from the TCL motion file
#     (--tclTsmFilter, default '*_MOT.tsm') whose time range covers the scan, if there is one
#   - everything is logged to outputBaseDir/<subject>/ptone_offline_batch--<YYYYMMDD-HHMMSS>.txt
# Raw data files whose MID already has a result in the output folder are skipped, unless --overwrite.
#
# Command line:
#   ptone-offline-batch <inputBaseDir> <outputBaseDir> --inputDirFilter 'ptoneH*'
#   (or python ptone/ptone_offline_batch.py ...)
# Python:
#   from ptone.ptone_offline_batch import main; main(['<inputBaseDir>', '<outputBaseDir>', ...])

import argparse
import fnmatch
import glob
import logging
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

ptoneDir = os.path.dirname(os.path.abspath(__file__))
repoDir  = os.path.dirname(ptoneDir)

logger = logging.getLogger('ptone_offline_batch')

# ----- Helpers -----------------------------------------------------------------------------------

def num_measurements(datPath):
    # Number of measurements in a Siemens raw data file.  VD/VE/XA files start with a multi-raid header:
    # a uint32 0, then the number of measurements (e.g. 1: AdjCoilSens, 2: the scan).  Older single
    # measurement (VB) files don't have one, so count as 1
    with open(datPath, 'rb') as f:
        header = f.read(8)
    if len(header) < 8:
        return 1
    first, numMeas = struct.unpack('<II', header)
    if first == 0 and 1 <= numMeas <= 64:
        return numMeas
    return 1

def mid_from_filename(path):
    # Siemens measurement ID from a raw data filename, e.g. 284 from 'meas_MID00284_FID14131_...dat'
    match = re.search(r'MID0*(\d+)', os.path.basename(path))
    return int(match.group(1)) if match else None

def find_files(baseDir, pattern):
    # Files under baseDir (recursively) whose names match pattern, sorted
    found = []
    for dirPath, dirNames, fileNames in os.walk(baseDir):
        dirNames.sort()
        found += [os.path.join(dirPath, f) for f in sorted(fileNames) if fnmatch.fnmatch(f, pattern)]
    return found

def results_for_mid(outputDir, mid):
    # Result files (.npz) for this MID in the output folder (pilottone.py names them <stem>--MID<nnnnn>-...)
    return set(glob.glob(os.path.join(outputDir, '*--MID%05d-*.npz' % mid)))

def siemens_to_ismrmrd_path():
    # siemens_to_ismrmrd from this Python's environment (or the conda environment a virtualenv is based on),
    # else from the PATH.  None if not found
    for binDir in (os.path.dirname(sys.executable), os.path.join(sys.prefix, 'bin'), os.path.join(sys.base_prefix, 'bin')):
        path = os.path.join(binDir, 'siemens_to_ismrmrd')
        if os.path.exists(path):
            return path
    return shutil.which('siemens_to_ismrmrd')

def run_logged(cmd, logPath, cwd=None):
    # Run a command with its output appended to the log; returns the exit code
    logger.info("Running: %s", ' '.join(cmd))
    for handler in logger.handlers:
        handler.flush()
    with open(logPath, 'a') as logFile:
        return subprocess.run(cmd, stdout=logFile, stderr=subprocess.STDOUT, cwd=cwd).returncode

def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
        except OSError:
            return False
    return True

def start_server(port, logPath, timeoutS=60):
    # Start an MRD server (main.py) on port, with its output appended to the log, and wait until it's listening
    if not port_is_free(port):
        raise RuntimeError("Port %d is already in use (another server or batch?)  Use --port" % port)
    logFile = open(logPath, 'a')
    server = subprocess.Popen([sys.executable, os.path.join(repoDir, 'main.py'), '-p', str(port)],
                              stdout=logFile, stderr=subprocess.STDOUT, cwd=repoDir)
    logFile.close()
    end = time.monotonic() + timeoutS
    while time.monotonic() < end:
        if server.poll() is not None:
            raise RuntimeError("MRD server exited with code %d.  See %s" % (server.returncode, logPath))
        with open(logPath, 'r', errors='replace') as f:
            if 'listening for data at' in f.read():
                logger.info("MRD server started on port %d (pid %d)", port, server.pid)
                return server
        time.sleep(0.2)
    stop_server(server)
    raise RuntimeError("MRD server didn't start within %d s.  See %s" % (timeoutS, logPath))

def stop_server(server):
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    logger.info("MRD server stopped")

def tcl_time_ranges(tclPaths):
    # (first, last) System Time (ms since midnight) of each TCL motion file; files that can't be read are left out
    sys.path.insert(0, repoDir)
    from ptone.tcl import read_tcl
    ranges = {}
    for path in tclPaths:
        try:
            times = read_tcl(path)['sys_time_ms_since_midnight'].values
            ranges[path] = (times.min(), times.max())
            logger.info("TCL motion file %s covers %s to %s", path, ms_to_time(times.min()), ms_to_time(times.max()))
        except Exception as e:
            logger.error("Could not read TCL motion file %s: %s", path, e)
    return ranges

def choose_tcl(npzPath, tclRanges):
    # The TCL motion file whose time range covers the scan in npzPath, or None
    timeMs = np.load(npzPath)['time_ms']
    for path, (first, last) in tclRanges.items():
        if first <= timeMs.min() and timeMs.max() <= last:
            return path
    return None

def ms_to_time(ms):
    return '%02d:%02d:%06.3f' % (ms // 3600000, ms % 3600000 // 60000, ms % 60000 / 1000)

# ----- Processing ------------------------------------------------------------------------------

def process_dat(datPath, subjectName, outputDir, tmpDir, port, config, overwrite, logPath):
    # Convert, analyse and return (status, new npz paths).  status: 'processed', 'skipped', 'no results' or 'failed'
    mid = mid_from_filename(datPath)
    if mid is None:
        logger.warning("No MID in the filename of %s, so it can't be checked for existing results", datPath)
    elif results_for_mid(outputDir, mid) and not overwrite:
        logger.info("Skipping %s: already processed (%s)", datPath, ', '.join(sorted(os.path.basename(p) for p in results_for_mid(outputDir, mid))))
        return 'skipped', []

    stem = os.path.splitext(os.path.basename(datPath))[0]
    h5Path = os.path.join(tmpDir, stem + '.h5')
    clientOutPath = os.path.join(tmpDir, stem + '--client.h5')
    before = results_for_mid(outputDir, mid) if mid is not None else set(glob.glob(os.path.join(outputDir, '*.npz')))
    try:
        converter = siemens_to_ismrmrd_path()
        if converter is None:
            logger.error("siemens_to_ismrmrd not found (it's installed in the mrd conda environment)")
            return 'failed', []
        numMeas = num_measurements(datPath)
        logger.info("Converting %s (measurement %d of %d)", datPath, numMeas, numMeas)
        start = time.monotonic()
        if run_logged([converter, '-f', datPath, '-z', str(numMeas), '-o', h5Path], logPath) != 0 \
                or not os.path.exists(h5Path):
            logger.error("Conversion failed: %s", datPath)
            return 'failed', []
        logger.info("Converted in %.0f s", time.monotonic() - start)

        start = time.monotonic()
        clientCmd = [sys.executable, os.path.join(repoDir, 'client.py'), '-p', str(port), '-c', config,
                     '--set', 'ptonePlot=false', '--set', 'outputFolder=' + outputDir,
                     '--set', 'outputFileStem=' + subjectName, '-o', clientOutPath, h5Path]
        if run_logged(clientCmd, logPath, cwd=repoDir) != 0:
            logger.error("Client failed: %s", datPath)
            return 'failed', []
        logger.info("Analysed in %.0f s", time.monotonic() - start)
    finally:
        for path in (h5Path, clientOutPath):
            if os.path.exists(path):
                os.remove(path)

    after = results_for_mid(outputDir, mid) if mid is not None else set(glob.glob(os.path.join(outputDir, '*.npz')))
    new = sorted(after - before)
    if not new:
        logger.warning("No results for %s (e.g. no lines analysed); see the scan's .txt log in %s", datPath, outputDir)
        return 'no results', []
    logger.info("Results: %s", ', '.join(os.path.basename(p) for p in new))
    return 'processed', new

def process_subject(subjectDir, outputBaseDir, args):
    subjectName = os.path.basename(subjectDir)
    outputDir = os.path.join(outputBaseDir, subjectName)
    tmpDir = os.path.join(args.tmpDir, subjectName) if args.tmpDir else outputDir
    os.makedirs(outputDir, exist_ok=True)
    os.makedirs(tmpDir, exist_ok=True)
    logPath = os.path.join(outputDir, 'ptone_offline_batch--%s.txt' % datetime.now().strftime('%Y%m%d-%H%M%S'))

    fileHandler = logging.FileHandler(logPath)
    fileHandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fileHandler)
    counts = {'processed': 0, 'skipped': 0, 'no results': 0, 'failed': 0, 'plotted': 0, 'plot failed': 0}
    subjectStart = time.monotonic()
    try:
        logger.info("===== Subject %s: %s -> %s", subjectName, subjectDir, outputDir)
        datPaths = find_files(subjectDir, args.measDatFilter)
        tclPaths = find_files(subjectDir, args.tclTsmFilter)
        logger.info("Found %d raw data file(s) matching '%s' and %d TCL motion file(s) matching '%s'",
                    len(datPaths), args.measDatFilter, len(tclPaths), args.tclTsmFilter)

        newResults = []
        if datPaths:
            server = start_server(args.port, logPath)
            try:
                for i, datPath in enumerate(datPaths):
                    logger.info("----- %s: file %d of %d: %s", subjectName, i + 1, len(datPaths), datPath)
                    try:
                        status, new = process_dat(datPath, subjectName, outputDir, tmpDir, args.port,
                                                  args.mrdClientConfig, args.overwrite, logPath)
                    except Exception:
                        logger.exception("Failed: %s", datPath)
                        status, new = 'failed', []
                    counts[status] += 1
                    newResults += new
            finally:
                stop_server(server)

        # Plot the new results, with head motion from the TCL file covering each scan
        tclRanges = tcl_time_ranges(tclPaths) if (newResults and tclPaths) else {}
        for npzPath in newResults:
            plotCmd = [sys.executable, os.path.join(ptoneDir, 'ptone_plot.py'), npzPath]
            try:
                tclPath = choose_tcl(npzPath, tclRanges)
            except Exception:
                logger.exception("Could not check TCL coverage for %s", npzPath)
                tclPath = None
            if tclPath is not None:
                plotCmd += ['--tcl', tclPath]
            elif tclPaths:
                logger.info("No TCL motion file covers %s; plotting without motion", os.path.basename(npzPath))
            if run_logged(plotCmd, logPath) == 0:
                counts['plotted'] += 1
            else:
                logger.error("Plot failed: %s", npzPath)
                counts['plot failed'] += 1
    except Exception:
        logger.exception("Subject %s failed", subjectName)
        counts['failed'] += 1
    finally:
        summary = ', '.join('%s %d' % (k, v) for k, v in counts.items())
        logger.info("===== Subject %s done in %.0f s: %s.  Log: %s", subjectName, time.monotonic() - subjectStart, summary, logPath)
        logger.removeHandler(fileHandler)
        fileHandler.close()
    return counts, summary

def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch-process previously acquired pilot tone data through pilottone.py")
    parser.add_argument('inputBaseDir', help='Directory containing one directory per subject')
    parser.add_argument('outputBaseDir', help='Results go to outputBaseDir/<subject>')
    parser.add_argument('--inputDirFilter', default='*', help="Subject directories to process, e.g. 'ptoneH*' (default: '*')")
    parser.add_argument('--mrdClientConfig', default='pilottone_offline',
                        help='Config passed to the client with -c (default: pilottone_offline)')
    parser.add_argument('--measDatFilter', default='meas_*.dat', help="Siemens raw data files (default: 'meas_*.dat')")
    parser.add_argument('--tclTsmFilter', default='*_MOT.tsm', help="TCL motion files (default: '*_MOT.tsm')")
    parser.add_argument('--port', type=int, default=9050, help='Port for the MRD server this script starts (default: 9050)')
    parser.add_argument('--tmpDir', default=None,
                        help='Where temporary .h5 files go, in a subdirectory per subject (default: the subject output directory)')
    parser.add_argument('--overwrite', action='store_true', help='Reprocess raw data files that already have results')
    args = parser.parse_args(argv)

    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    if not os.path.isdir(args.inputBaseDir):
        logger.error("Input directory not found: %s", args.inputBaseDir)
        return 1
    subjects = sorted(d for d in os.listdir(args.inputBaseDir)
                      if fnmatch.fnmatch(d, args.inputDirFilter) and os.path.isdir(os.path.join(args.inputBaseDir, d)))
    if not subjects:
        logger.error("No directories in %s match '%s'", args.inputBaseDir, args.inputDirFilter)
        return 1
    logger.info("Processing %d subject(s): %s", len(subjects), ', '.join(subjects))

    summaries = []
    anyFailed = False
    for subjectName in subjects:
        counts, summary = process_subject(os.path.join(args.inputBaseDir, subjectName), args.outputBaseDir, args)
        summaries.append((subjectName, summary))
        anyFailed = anyFailed or counts['failed'] > 0 or counts['plot failed'] > 0

    print("\nSummary:")
    for subjectName, summary in summaries:
        print("  %-40s %s" % (subjectName, summary))
    return 1 if anyFailed else 0

if __name__ == '__main__':
    sys.exit(main())
