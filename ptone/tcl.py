# Read TCL (TracSuite) head motion data and match it to scan lines by time.
#
# Adapted from kstream's kstream/tcl_sync.py (read_tcl and the nearest-neighbour matching in
# sync_meas_tcl).  Changes: only the motion file (*_MOT.tsm) is read.  Its 'System Time' (the scanner
# clock) is the only time the matching uses; kstream also merges in the timing file (*_TIM.tst), but its
# 'Remote Time' is the same as 'System Time' and its other columns aren't used.  Lines are matched by
# their scanner timestamps (ms since midnight) instead of a twix mdb list.
#
# Also: choosing the motion file that covers a scan (time_range, choose_tcl), and adding the matched
# motion to a pilottone.py results file (add_tcl_to_npz).

import logging
import os

import numpy as np
import pandas

def read_tcl(motPath):
    """
    Read a TracSuite motion file (*_MOT.tsm).

    Returns a DataFrame with the file's columns ('Point Cloud Number', 'System Time', 'Tx', 'Ty', 'Tz',
    'Rx', 'Ry', 'Rz', '3D motion') and 'sys_time_ms_since_midnight': 'System Time' in milliseconds
    since midnight.
    """
    if not os.path.isfile(motPath):
        raise FileNotFoundError("TCL motion file not found: " + motPath)

    # The column header is the line starting with 'Point Cloud Number'
    headerRow = None
    with open(motPath, 'r') as f:
        for i, line in enumerate(f):
            if line.strip().startswith('Point Cloud Number'):
                headerRow = i
                break
    if headerRow is None:
        raise ValueError("Could not find header row in " + motPath)

    tclDf = pandas.read_csv(motPath, skiprows=headerRow, sep=r'\s{2,}', engine='python')

    td = pandas.to_timedelta(tclDf['System Time'])
    tclDf['sys_time_ms_since_midnight'] = (td.dt.total_seconds() * 1e3).astype(int)
    return tclDf

def match_tcl(tclDf, timeMs):
    """
    Nearest TCL sample for each scan line.

    timeMs: the lines' scanner timestamps in ms since midnight (the npz 'time_ms').
    Returns a DataFrame with one row per line: 'Point Cloud Number', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz',
    '3D motion', and '3D motion framewise' (absolute change in '3D motion' from the previous line).
    """
    tclTimes = tclDf['sys_time_ms_since_midnight'].values
    timeMs = np.asarray(timeMs, dtype=float)

    if (timeMs.min() < tclTimes.min()) or (timeMs.max() > tclTimes.max()):
        logging.warning("Scan lines (%.0f to %.0f ms since midnight) extend outside the TCL data "
                        "(%.0f to %.0f ms); those lines are matched to the nearest end of the TCL data",
                        timeMs.min(), timeMs.max(), tclTimes.min(), tclTimes.max())

    # searchsorted gives the insertion point; check whether the previous sample is closer
    idxs = np.searchsorted(tclTimes, timeMs)
    idxs = np.clip(idxs, 1, len(tclTimes) - 1)
    leftDiff = np.abs(timeMs - tclTimes[idxs - 1])
    rightDiff = np.abs(timeMs - tclTimes[idxs])
    idxs = np.where(leftDiff <= rightDiff, idxs - 1, idxs)

    tclCols = ['Point Cloud Number', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', '3D motion']
    matched = tclDf.iloc[idxs][tclCols].reset_index(drop=True)
    matched['3D motion framewise'] = matched['3D motion'].diff().fillna(0).abs()
    return matched

def time_range(tclDf):
    """(first, last) 'System Time' of a read_tcl() DataFrame, in ms since midnight."""
    times = tclDf['sys_time_ms_since_midnight'].values
    return times.min(), times.max()

def covers(tclRange, timeMs):
    """Whether a TCL time range (first, last) covers all of timeMs (ms since midnight)."""
    first, last = tclRange
    return first <= np.min(timeMs) and np.max(timeMs) <= last

def choose_tcl(timeMs, tclRanges):
    """
    The TCL motion file that covers a scan, or None.

    timeMs:    the scan's line timestamps in ms since midnight (the npz 'time_ms')
    tclRanges: {motPath: time_range(read_tcl(motPath))} for the candidate motion files
    """
    for path, tclRange in tclRanges.items():
        if covers(tclRange, timeMs):
            return path
    return None

# Arrays add_tcl_to_npz() adds to a results file: npz key -> match_tcl() column
NPZ_TCL_KEYS = {
    'tcl_Tx':                  'Tx',
    'tcl_Ty':                  'Ty',
    'tcl_Tz':                  'Tz',
    'tcl_Rx':                  'Rx',
    'tcl_Ry':                  'Ry',
    'tcl_Rz':                  'Rz',
    'tcl_3d_motion':           '3D motion',
    'tcl_3d_motion_framewise': '3D motion framewise',
    'tcl_point_cloud_number':  'Point Cloud Number',
}

def add_tcl_to_npz(npzPath, motPath, tclDf=None):
    """
    Add TCL head motion, matched to each analysed line, to a pilottone.py results file (.npz).

    The file is rewritten with all its existing arrays plus the NPZ_TCL_KEYS arrays (one value per line,
    matched by 'time_ms' with match_tcl()) and 'tcl_file' (motPath).  Any tcl_ arrays already in the file
    are replaced.  The file is written to a temporary file first and then renamed, so an interruption
    can't leave it half-written.

    Raises ValueError, leaving the file unchanged, if the motion file doesn't cover the whole scan: lines
    outside it would be matched to the nearest end of the motion data, which would be wrong.

    tclDf: read_tcl(motPath), if already read (e.g. when adding the same file to several scans).
    """
    if tclDf is None:
        tclDf = read_tcl(motPath)
    with np.load(npzPath) as d:
        arrays = {k: d[k] for k in d.files if not k.startswith('tcl_')}
    if not covers(time_range(tclDf), arrays['time_ms']):
        raise ValueError("TCL motion file %s doesn't cover the whole scan in %s, so its motion wasn't added"
                         % (motPath, npzPath))
    matched = match_tcl(tclDf, arrays['time_ms'])
    for key, col in NPZ_TCL_KEYS.items():
        arrays[key] = matched[col].values
    arrays['tcl_file'] = np.array(os.path.abspath(motPath))

    tmpPath = npzPath + '.tmp'
    with open(tmpPath, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmpPath, npzPath)
