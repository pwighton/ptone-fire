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

# ----- add_tcl_to_npz ----------------------------------------------------------------------------

def make_results(path, startMs, numLines=30, periodMs=150):
    # A minimal results file like pilottone.py's, with a string entry as well as arrays
    rng = np.random.default_rng(1)
    np.savez(path, time_ms=startMs + periodMs * np.arange(numLines), relative_phase=rng.random((numLines, 4)),
             quality=rng.random(numLines), settings=np.array('{"refChanIdx": 0}'))
    return str(path)

def test_add_tcl_to_npz(tmp_path):
    from ptone.tcl import add_tcl_to_npz, NPZ_TCL_KEYS
    motPath = str(tmp_path / 'session_120616_MOT.tsm')
    startMs = write_tcl(motPath, numSamples=100)
    npzPath = make_results(tmp_path / 'results.npz', startMs + 200)
    with np.load(npzPath) as d:
        before = {k: d[k] for k in d.files}

    add_tcl_to_npz(npzPath, motPath)

    expected = match_tcl(read_tcl(motPath), before['time_ms'])
    with np.load(npzPath) as d:
        # Original contents unchanged
        for k, v in before.items():
            assert np.array_equal(d[k], v), k
        # TCL arrays added, one value per line, as match_tcl() gives
        for key, col in NPZ_TCL_KEYS.items():
            assert d[key].shape == before['time_ms'].shape, key
            assert np.allclose(d[key], expected[col].values), key
        assert str(d['tcl_file']) == os.path.abspath(motPath)
        assert set(d.files) == set(before) | set(NPZ_TCL_KEYS) | {'tcl_file'}
    assert not os.path.exists(npzPath + '.tmp')

def test_add_tcl_to_npz_replaces_existing(tmp_path):
    from ptone.tcl import add_tcl_to_npz
    first = str(tmp_path / 'first_MOT.tsm'); second = str(tmp_path / 'second_MOT.tsm')
    startMs = write_tcl(first, numSamples=100)
    write_tcl(second, startTime='11:29:38.541', numSamples=100, periodMs=50)    # Same start, different times
    npzPath = make_results(tmp_path / 'results.npz', startMs + 200, numLines=10)
    add_tcl_to_npz(npzPath, first)
    add_tcl_to_npz(npzPath, second, tclDf=read_tcl(second))
    with np.load(npzPath) as d:
        assert str(d['tcl_file']) == os.path.abspath(second)
        assert np.allclose(d['tcl_3d_motion'], match_tcl(read_tcl(second), d['time_ms'])['3D motion'].values)

def test_add_tcl_to_npz_requires_coverage(tmp_path):
    # A motion file that doesn't cover the whole scan isn't added, and the file is left unchanged
    from ptone.tcl import add_tcl_to_npz
    motPath = str(tmp_path / 'session_120616_MOT.tsm')
    startMs = write_tcl(motPath, numSamples=100)                                    # 10 s
    npzPath = make_results(tmp_path / 'results.npz', startMs + 8000, numLines=30)   # Runs past the end
    before = open(npzPath, 'rb').read()
    with pytest.raises(ValueError, match="doesn't cover the whole scan"):
        add_tcl_to_npz(npzPath, motPath)
    assert open(npzPath, 'rb').read() == before

# ----- Choosing the motion file covering a scan ----------------------------------------------------

def test_choose_tcl_by_time_range(tmp_path):
    from ptone.tcl import time_range, choose_tcl
    morning = str(tmp_path / 'morning_MOT.tsm')
    afternoon = str(tmp_path / 'afternoon_MOT.tsm')
    morningStart = write_tcl(morning, startTime='09:00:00.000', numSamples=1000)       # 100 s
    afternoonStart = write_tcl(afternoon, startTime='14:00:00.000', numSamples=1000)
    ranges = {p: time_range(read_tcl(p)) for p in (morning, afternoon)}
    assert ranges[morning] == (morningStart, morningStart + 99900)

    lines = lambda startMs: startMs + np.arange(0, 50000, 100)                          # 50 s of lines
    assert choose_tcl(lines(afternoonStart + 10000), ranges) == afternoon
    assert choose_tcl(lines(morningStart), ranges) == morning
    assert choose_tcl(lines(afternoonStart + 80000), ranges) is None                   # Runs past the end
    assert choose_tcl(lines(morningStart - 3600000), ranges) is None
    assert choose_tcl(lines(morningStart), {}) is None
