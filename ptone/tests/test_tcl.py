# Tests for ptone/tcl.py
#
# The real-data test looks for the TracSuite files and the April TSE in $PTONE_TEST_DATA, or
# ../pilot-tone-test-data relative to the repository root, and is skipped if they're not found.

import logging
import os

import numpy as np
import pytest

from ptone.tcl import read_tcl, match_tcl

TRACSUITE_HEADER = '''################################################################################
#                                  TracSuite                                   #
################################################################################

TracSuite Motion File

Reference frame: 2948

'''

def write_tcl(motPath, startTime='11:29:38.541', numSamples=20, periodMs=100):
    # Synthetic TracSuite motion file (*_MOT.tsm) with '3D motion' = 0.1 mm per sample.  Returns the
    # first sample's time in ms since midnight
    h, m, s = startTime.split(':')
    startMs = (int(h) * 3600 + int(m) * 60 + float(s)) * 1000
    def fmt(ms):
        return '%02d:%02d:%06.3f' % (ms // 3600000, ms % 3600000 // 60000, ms % 60000 / 1000)
    with open(motPath, 'w') as f:
        f.write(TRACSUITE_HEADER)
        f.write('Point Cloud Number       System Time                Tx                Ty                Tz'
                '                Rx                Ry                Rz         3D motion\n')
        for i in range(numSamples):
            f.write('%18d      %s  %16.6f  %16.6f  %16.6f  %16.6f  %16.6f  %16.6f  %16.6f\n'
                    % (3044 + i, fmt(startMs + i * periodMs), 0.01 * i, 0, 0, 0, 0, 0, 0.1 * i))
    return startMs

def test_read_tcl(tmp_path):
    motPath = str(tmp_path / 'session_120616_MOT.tsm')
    startMs = write_tcl(motPath)
    df = read_tcl(motPath)
    assert len(df) == 20
    assert df['sys_time_ms_since_midnight'].iloc[0] == int(startMs)
    assert df['3D motion'].iloc[3] == pytest.approx(0.3)

def test_read_tcl_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_tcl(str(tmp_path / 'nonexistent_MOT.tsm'))

def test_match_tcl_nearest_and_framewise(tmp_path):
    motPath = str(tmp_path / 'session_120616_MOT.tsm')
    startMs = write_tcl(motPath)
    df = read_tcl(motPath)
    # Lines at +0, +140, +160 and +960 ms: nearest samples 0, 1, 2 and 10 (samples every 100 ms)
    matched = match_tcl(df, startMs + np.array([0, 140, 160, 960]))
    assert list(matched['Point Cloud Number']) == [3044, 3045, 3046, 3054]
    assert matched['3D motion framewise'].tolist() == pytest.approx([0, 0.1, 0.1, 0.8])

def test_match_tcl_warns_outside_range(tmp_path, caplog):
    motPath = str(tmp_path / 'session_120616_MOT.tsm')
    startMs = write_tcl(motPath)
    df = read_tcl(motPath)
    with caplog.at_level(logging.WARNING):
        matched = match_tcl(df, startMs + np.array([-5000, 0]))
    assert 'outside the TCL data' in caplog.text
    assert list(matched['Point Cloud Number']) == [3044, 3044]

# ----- Real data ------------------------------------------------------------------------------

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataDir = os.environ.get('PTONE_TEST_DATA', os.path.join(repoDir, '..', 'pilot-tone-test-data'))
realMot = os.path.join(dataDir, '2026-04-29_ptoneH20260429', '2026-04-29_ptoneH20260429_120616_MOT.tsm')
realTse = os.path.join(dataDir, 'ptoneH20260429--meas_MID00284_FID14131_t2_tse_tra_dark_fluid__m_pt.h5')

def test_real_tcl_covers_april_tse(caplog):
    if not (os.path.exists(realMot) and os.path.exists(realTse)):
        pytest.skip('Real TCL data or April TSE not found in ' + dataDir)
    import h5py
    df = read_tcl(realMot)
    with h5py.File(realTse, 'r') as f:
        timeMs = f['dataset']['data']['head']['acquisition_time_stamp'][1:].astype(float) * 2.5  # Skip the noise line
    with caplog.at_level(logging.WARNING):
        matched = match_tcl(df, timeMs)
    assert 'outside the TCL data' not in caplog.text
    # The TCL camera samples at ~36 Hz, so every line is within a sample period of its match
    matchedMs = df.set_index('Point Cloud Number').loc[matched['Point Cloud Number'], 'sys_time_ms_since_midnight'].values
    assert np.max(np.abs(matchedMs - timeMs)) < 50
