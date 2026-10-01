# Read TCL (TracSuite) head motion data and match it to scan lines by time.
#
# Adapted from kstream's kstream/tcl_sync.py (read_tcl and the nearest-neighbour matching in
# sync_meas_tcl).  Changes: only the motion file (*_MOT.tsm) is read.  Its 'System Time' (the scanner
# clock) is the only time the matching uses; kstream also merges in the timing file (*_TIM.tst), but its
# 'Remote Time' is the same as 'System Time' and its other columns aren't used.  Lines are matched by
# their scanner timestamps (ms since midnight) instead of a twix mdb list.

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
