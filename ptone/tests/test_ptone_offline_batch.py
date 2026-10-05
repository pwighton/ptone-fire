# Tests for ptone/ptone_offline_batch.py
#
# The real-data tests look for data in $PTONE_TEST_DATA (default ../pilot-tone-test-data relative to the
# repository root) and are skipped if it's not found.  The end-to-end test takes a minute or two.

import glob
import json
import os
import socket
import struct

import numpy as np
import pytest

from ptone import ptone_offline_batch as batch
from ptone.tests.test_tcl import write_tcl

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataDir = os.environ.get('PTONE_TEST_DATA', os.path.join(repoDir, '..', 'pilot-tone-test-data'))
aprilDat = os.path.join(dataDir, 'ptoneH20260429--meas_MID00284_FID14131_t2_tse_tra_dark_fluid__m_pt.dat')
aprilMot = os.path.join(dataDir, '2026-04-29_ptoneH20260429', '2026-04-29_ptoneH20260429_120616_MOT.tsm')

def free_port():
    with socket.socket() as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]

# ----- Raw data files ----------------------------------------------------------------------------

@pytest.mark.parametrize('numMeas', [1, 2, 3])
def test_num_measurements_multiraid(tmp_path, numMeas):
    path = tmp_path / 'meas.dat'
    path.write_bytes(struct.pack('<II', 0, numMeas) + b'\0' * 100)
    assert batch.num_measurements(str(path)) == numMeas

def test_num_measurements_single_measurement_file(tmp_path):
    # VB files have no multi-raid header: they start with the header length
    path = tmp_path / 'meas.dat'
    path.write_bytes(struct.pack('<II', 10240, 12345) + b'\0' * 100)
    assert batch.num_measurements(str(path)) == 1

def test_num_measurements_real():
    if not os.path.exists(aprilDat):
        pytest.skip('Not found: ' + aprilDat)
    assert batch.num_measurements(aprilDat) == 2     # AdjCoilSens + the TSE

@pytest.mark.parametrize('name, mid', [
    ('meas_MID00284_FID14131_t2_tse_tra_dark_fluid.dat', 284),
    ('ptoneH20260429--meas_MID01104_FID99032_gre4mm_sag.dat', 1104),
    ('meas_FID14131_no_mid.dat', None),
])
def test_mid_from_filename(name, mid):
    assert batch.mid_from_filename('/some/dir/' + name) == mid

def test_already_processed_is_skipped(tmp_path):
    outputDir = tmp_path / 'out'
    outputDir.mkdir()
    (outputDir / 'ptoneH20260429--MID00284-t2_tse--20261002-120000-000.npz').write_bytes(b'')
    status, new = batch.process_dat('/nonexistent/meas_MID00284_FID1_x.dat', 'ptoneH20260429', str(outputDir),
                                    str(tmp_path), 0, 'pilottone_offline', False, str(tmp_path / 'log.txt'))
    assert (status, new) == ('skipped', [])

# ----- TCL motion --------------------------------------------------------------------------------

def test_tcl_time_ranges_skips_unreadable(tmp_path):
    pytest.importorskip('pandas')
    good = str(tmp_path / 'good_MOT.tsm')
    startMs = write_tcl(good, startTime='09:00:00.000', numSamples=1000)
    bad = tmp_path / 'bad_MOT.tsm'
    bad.write_text('not a TracSuite file')
    assert batch.tcl_time_ranges([good, str(bad)]) == {good: (startMs, startMs + 99900)}

def make_plottable_result(path, startMs):
    # A results file with everything ptone_plot.py needs
    from ptone.tests.test_ptone_plot import make_npz
    return make_npz(path, startMs=startMs)

def test_plot_with_tcl(tmp_path):
    # Results covered by a TCL file get its motion (saved by ptone_plot.py --add-tcl); others are plotted
    # without it, or, with onlyWithTcl, not at all
    pytest.importorskip('pandas')
    motPath = str(tmp_path / 'session_MOT.tsm')
    startMs = write_tcl(motPath, startTime='09:00:00.000', numSamples=1000)          # 100 s
    covered = make_plottable_result(tmp_path / 'covered.npz', startMs + 1000)
    outside = make_plottable_result(tmp_path / 'outside.npz', startMs + 3600000)
    logPath = str(tmp_path / 'log.txt')

    counts = {'tcl added': 0, 'no tcl': 0, 'plotted': 0, 'plot failed': 0}
    batch.plot_with_tcl([covered, outside], [motPath], logPath, counts)
    assert counts == {'tcl added': 1, 'no tcl': 1, 'plotted': 2, 'plot failed': 0}
    assert str(np.load(covered)['tcl_file']) == motPath
    assert not any(k.startswith('tcl_') for k in np.load(outside).files)
    assert os.path.exists(covered.replace('.npz', '.png')) and os.path.exists(outside.replace('.npz', '.png'))

    os.remove(outside.replace('.npz', '.png'))
    counts = {'tcl added': 0, 'no tcl': 0, 'plotted': 0, 'plot failed': 0}
    batch.plot_with_tcl([outside], [], logPath, counts, onlyWithTcl=True)
    assert counts == {'tcl added': 0, 'no tcl': 1, 'plotted': 0, 'plot failed': 0}
    assert not os.path.exists(outside.replace('.npz', '.png'))

def test_add_tcl_only(tmp_path):
    # --addTclOnly adds TCL data to existing results without processing raw data (no server), and re-plots
    # only the results it added TCL data to
    pytest.importorskip('pandas')
    inputDir = tmp_path / 'input'
    subjectDir = inputDir / 'ptoneH20260429'
    subjectDir.mkdir(parents=True)
    startMs = write_tcl(str(subjectDir / 'session_MOT.tsm'), startTime='09:00:00.000', numSamples=1000)
    (subjectDir / 'meas_MID00284_FID1_x.dat').write_bytes(b'not a real raw data file')
    subjectOut = tmp_path / 'output' / 'ptoneH20260429'
    subjectOut.mkdir(parents=True)
    covered = make_plottable_result(subjectOut / 'ptoneH20260429--MID00284-x--20261002-120000-000.npz', startMs + 1000)
    outside = make_plottable_result(subjectOut / 'ptoneH20260429--MID00285-y--20261002-120100-000.npz', startMs + 3600000)

    assert batch.main([str(inputDir), str(tmp_path / 'output'), '--addTclOnly', '--port', '1']) == 0
    with np.load(covered) as d:
        assert str(d['tcl_file']) == str(subjectDir / 'session_MOT.tsm')
    assert os.path.exists(covered.replace('.npz', '.png'))                  # Re-plotted with its TCL motion
    assert not os.path.exists(outside.replace('.npz', '.png'))              # No TCL data, so not re-plotted
    log = open(glob.glob(str(subjectOut / 'ptone_offline_batch--*.txt'))[0]).read()
    assert 'tcl added 1' in log and 'no tcl 1' in log and 'plotted 1' in log and 'plot failed 0' in log
    assert 'processed 0' in log and 'MRD server' not in log

# ----- End to end --------------------------------------------------------------------------------

def test_end_to_end(tmp_path):
    if not (os.path.exists(aprilDat) and os.path.exists(aprilMot)):
        pytest.skip('April TSE .dat or TCL motion file not found in ' + dataDir)
    pytest.importorskip('pandas')

    # Input tree: a human subject with the April TSE and its TCL data, and a phantom subject the filter excludes
    inputDir = tmp_path / 'input'
    subjectDir = inputDir / 'ptoneH20260429'
    (subjectDir / 'raw').mkdir(parents=True)
    (subjectDir / 'tcl').mkdir()
    os.symlink(aprilDat, subjectDir / 'raw' / 'meas_MID00284_FID14131_t2_tse_tra_dark_fluid__m_pt.dat')
    os.symlink(aprilMot, subjectDir / 'tcl' / '2026-04-29_ptoneH20260429_120616_MOT.tsm')
    (inputDir / 'ptone20251205').mkdir()
    outputDir = tmp_path / 'output'
    args = [str(inputDir), str(outputDir), '--inputDirFilter', 'ptoneH*', '--port', str(free_port())]

    assert batch.main(args) == 0
    subjectOut = outputDir / 'ptoneH20260429'
    assert not (outputDir / 'ptone20251205').exists()
    npz = glob.glob(str(subjectOut / 'ptoneH20260429--MID00284-*.npz'))
    assert len(npz) == 1
    assert os.path.exists(npz[0].replace('.npz', '.png'))
    assert not glob.glob(str(subjectOut / '*.h5'))                          # Temporary files deleted
    settings = json.loads(str(np.load(npz[0])['settings']))
    assert settings['outputFileStem'] == 'ptoneH20260429' and settings['ptonePlot'] is False
    logs = glob.glob(str(subjectOut / 'ptone_offline_batch--*.txt'))
    assert len(logs) == 1
    log = open(logs[0]).read()
    assert 'processed 1' in log and 'tcl added 1' in log
    with np.load(npz[0]) as d:                                              # TCL head motion added
        assert str(d['tcl_file']).endswith('2026-04-29_ptoneH20260429_120616_MOT.tsm')
        assert d['tcl_3d_motion'].shape == d['time_ms'].shape

    # Second run: already processed, so skipped
    assert batch.main(args) == 0
    assert len(glob.glob(str(subjectOut / '*.npz'))) == 1
    logs = sorted(glob.glob(str(subjectOut / 'ptone_offline_batch--*.txt')))
    assert 'skipped 1' in open(logs[-1]).read()
