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

# ----- TCL choice --------------------------------------------------------------------------------

def test_choose_tcl_by_time_range(tmp_path):
    pytest.importorskip('pandas')
    morning = str(tmp_path / 'morning_MOT.tsm')
    afternoon = str(tmp_path / 'afternoon_MOT.tsm')
    morningStart = write_tcl(morning, startTime='09:00:00.000', numSamples=1000)     # 100 s
    afternoonStart = write_tcl(afternoon, startTime='14:00:00.000', numSamples=1000)
    ranges = batch.tcl_time_ranges([morning, afternoon])

    def npz_at(name, startMs):
        path = str(tmp_path / name)
        np.savez(path, time_ms=startMs + np.arange(0, 50000, 100))   # 50 s of lines
        return path
    assert batch.choose_tcl(npz_at('a.npz', afternoonStart + 10000), ranges) == afternoon
    assert batch.choose_tcl(npz_at('m.npz', morningStart), ranges) == morning
    assert batch.choose_tcl(npz_at('late.npz', afternoonStart + 80000), ranges) is None   # Runs past the end
    assert batch.choose_tcl(npz_at('none.npz', morningStart - 3600000), ranges) is None

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
    assert '--tcl' in log and '_MOT.tsm' in log                             # Plotted with head motion
    assert 'processed 1' in log

    # Second run: already processed, so skipped
    assert batch.main(args) == 0
    assert len(glob.glob(str(subjectOut / '*.npz'))) == 1
    logs = sorted(glob.glob(str(subjectOut / 'ptone_offline_batch--*.txt')))
    assert 'skipped 1' in open(logs[-1]).read()
