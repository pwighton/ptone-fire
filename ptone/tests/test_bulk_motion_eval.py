# Tests for ptone/bulk_motion_eval.py, on small synthetic results files with TCL data

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ptone import bulk_motion_eval as ev

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def make_result(path, protocol='t2_fl2d_tra_hemo--nm-pt', durationS=60.0, spacingMs=40.0, numChan=6,
                moveAtS=None, phaseStep=0.05, mmStep=2.0, trackerLagS=0.4, noise=0.003, seed=0):
    # A results file like pilottone.py's, with TCL data.  At moveAtS (if set) the head moves: the relative
    # phase steps by phaseStep on every channel but the reference, and the tracker reports a mmStep
    # translation trackerLagS later (it lags the tone)
    rng = np.random.default_rng(seed)
    t = np.arange(0, durationS * 1000, spacingMs) + 40000000.0
    ph = rng.uniform(-1, 1, numChan) + noise * rng.standard_normal((len(t), numChan))
    pos = np.zeros((len(t), 3)); rot = np.zeros((len(t), 3))
    if moveAtS is not None:
        ph[t - t[0] >= moveAtS * 1000, 1:] += phaseStep
        pos[t - t[0] >= (moveAtS + trackerLagS) * 1000, 0] += mmStep
    ph[:, 0] = 0.0
    header = ('<ismrmrdHeader><measurementInformation><protocolName>%s</protocolName></measurementInformation>'
              '<sequenceParameters><TR>597.0</TR></sequenceParameters></ismrmrdHeader>' % protocol)
    np.savez(path, time_ms=t, quality=np.full(len(t), 0.97), relative_phase=ph, relative_amplitude=np.ones_like(ph),
             settings=np.array(json.dumps({'refChanIdx': 0, 'ptoneQualityThreshold': 0.5})), mrd_header=np.array(header),
             tcl_Tx=pos[:, 0], tcl_Ty=pos[:, 1], tcl_Tz=pos[:, 2], tcl_Rx=rot[:, 0], tcl_Ry=rot[:, 1], tcl_Rz=rot[:, 2])
    return str(path)

# ----- Labels ------------------------------------------------------------------------------------

@pytest.mark.parametrize('protocol, group, kind', [
    ('t2_tse_tra_dark-fluid--m-pt', 'TSE', 'm'),
    ('t2_tse_tra_dark-fluid_ARIA--nm-pt', 'TSE', 'nm'),
    ('t2_fl2d_tra_hemo--nm-pt', 'FLASH', 'nm'),
    ('TRA_SWI--m-pt', 'SWI', 'm'),
    ('t1_mprage--nm-pt', 'MPRAGE', 'nm'),
    ('gre4mm_sag', 'gre4mm_sag', '?'),
    ('ep2d_bold--m', 'ep2d_bold', 'm'),
])
def test_labels(protocol, group, kind):
    assert ev.sequence_group(protocol) == group
    assert ev.motion_kind(protocol) == kind

def test_auc():
    assert ev.auc([3, 4], [1, 2]) == 1.0
    assert ev.auc([1, 2], [3, 4]) == 0.0
    assert ev.auc([1, 2], [1, 2]) == 0.5               # Ties count half
    assert ev.auc([2], [1, 3]) == 0.5
    assert np.isnan(ev.auc([], [1]))

# ----- Jenkinson's RMS deviation ---------------------------------------------------------------

def test_jenkinson_rms_translation_and_identity():
    zero = (0, 0, 0)
    assert ev.jenkinson_rms(zero, zero, zero, zero) == 0.0
    assert ev.jenkinson_rms(zero, zero, (3, 4, 0), zero) == pytest.approx(5.0)       # Pure translation: its length
    pose = ((1, 2, 3), (2, -1, 0.5))
    assert ev.jenkinson_rms(*pose, *pose) == pytest.approx(0.0, abs=1e-9)            # No movement

def test_jenkinson_rms_rotation():
    # Rotation by theta about z: tr(A^T A) = 4 (1 - cos theta), so RMS = sqrt(r^2 / 5 * 4 (1 - cos theta)) about
    # the centre, plus the centre's own displacement |A c| = 2 |c| sin(theta / 2) for c in the x-y plane
    zero = (0, 0, 0)
    theta = np.radians(1.0)
    about0 = np.sqrt(60 ** 2 / 5 * 4 * (1 - np.cos(theta)))
    assert ev.jenkinson_rms(zero, zero, zero, (0, 0, 1.0)) == pytest.approx(about0)
    assert about0 == pytest.approx(60 * theta * np.sqrt(2 / 5), rel=1e-4)           # ~0.66 mm per degree
    centred = np.sqrt(about0 ** 2 + (2 * 90 * np.sin(theta / 2)) ** 2)
    assert ev.jenkinson_rms(zero, zero, zero, (0, 0, 1.0), centre=(0, 90, 0)) == pytest.approx(centred)
    assert ev.jenkinson_rms(zero, zero, zero, (0, 0, 1.0), radius=30) == pytest.approx(about0 / 2)

def test_jenkinson_rms_relative_and_symmetric():
    # Between two poses it depends only on the movement between them, and is the same either way round
    a = ((5, -3, 2), (3, 1, -2)); b = ((6, -2, 2.5), (3.5, 0.5, -1))
    assert ev.jenkinson_rms(*a, *b) == pytest.approx(ev.jenkinson_rms(*b, *a))
    assert ev.jenkinson_rms(*a, *b) > 0.5

# ----- Scoring against the tracker ---------------------------------------------------------------

def test_score_scan_moved_window(tmp_path):
    path = make_result(tmp_path / 'a--MID00001-x.npz', moveAtS=31.0)
    rows = ev.score_scan(ev.load_scan(path), params={'minWindowS': 3})
    moved = [r for r in rows if r['trackerMm'] > 1]
    # One window shows the move (the 30-33 s window: more than half of it is after the move)
    assert len(moved) == 1 and moved[0]['windowStartS'] == pytest.approx(30, abs=0.05)
    assert moved[0]['trackerMm'] == pytest.approx(2.0)
    assert moved[0]['score'] > 10 * max(r['score'] for r in rows if r is not moved[0])

def test_tracker_lag(tmp_path):
    # The tone moves at 31.4 s (53% of the 30-33 s window after it, so that window's score shows it); the
    # tracker reports it 1 s later, at 32.4 s (20% of the window after it).  Shifted by the lag, the tracker
    # puts the move in the same window as the tone; unshifted, in the next one
    path = make_result(tmp_path / 'a--MID00001-x.npz', moveAtS=31.4, mmStep=2.0, trackerLagS=1.0)
    scan = ev.load_scan(path)
    aligned = ev.score_scan(scan, params={'minWindowS': 3}, lagS=1.0)
    unaligned = ev.score_scan(scan, params={'minWindowS': 3}, lagS=0)
    best = lambda rows: max(rows, key=lambda r: r['score'])
    assert best(aligned)['trackerMm'] == pytest.approx(2.0)
    assert best(unaligned)['trackerMm'] < 0.1         # The tracker's move lands in the next window

def test_files_without_tcl_are_skipped(tmp_path):
    path = make_result(tmp_path / 'a--MID00001-x.npz')
    d = dict(np.load(path))
    for k in [k for k in d if k.startswith('tcl_')]:
        del d[k]
    np.savez(tmp_path / 'b--MID00002-x.npz', **d)
    windows, scans = ev.evaluate([str(tmp_path)], params={'minWindowS': 3})
    assert set(s['session'] for s in scans) == {'a'}

def test_evaluate_and_summarize(tmp_path):
    # Two sessions: a still scan and a motion scan each
    for i, s in enumerate(('s1', 's2')):
        make_result(tmp_path / ('%s--MID00001-still.npz' % s), protocol='TRA_SWI--nm-pt', seed=10 * i)
        make_result(tmp_path / ('%s--MID00002-move.npz' % s), protocol='TRA_SWI--m-pt', moveAtS=31.0, seed=10 * i + 1)
    windows, scans = ev.evaluate([str(tmp_path)], params={'minWindowS': 3})
    assert len(scans) == 4 and all(s['firstScoreS'] == pytest.approx(6.0, abs=0.05) for s in scans)
    summary = {r['group']: r for r in ev.summarize(windows, scans)}
    assert set(summary) == {'SWI', 'pooled'}
    swi = summary['SWI']
    assert swi['movedWindows'] == 2 and swi['aucMoved'] == 1.0 and swi['auc_2_inf'] == 1.0
    assert swi['stillMedian'] < 0.001
    # ~18 still windows per session, so a 2% target lets through at most ~1 of them
    assert swi['losoFaMedian'] <= 0.1 and swi['losoCaught_ge2'] == 1.0
    assert summary['pooled']['scans'] == 4
    levels = ev.session_levels(windows)
    assert [(l['session'], l['group']) for l in levels] == [('s1', 'SWI'), ('s2', 'SWI')]

def test_exclude_sessions(tmp_path):
    make_result(tmp_path / 's1--MID00001-a.npz', protocol='TRA_SWI--nm-pt')
    make_result(tmp_path / 's2--MID00001-a.npz', protocol='TRA_SWI--nm-pt')
    windows, scans = ev.evaluate([str(tmp_path)], params={'minWindowS': 3}, excludeSessions=['s2'])
    assert set(s['session'] for s in scans) == {'s1'} and set(w['session'] for w in windows) == {'s1'}

def test_context_groups_left_out_of_pooled(tmp_path):
    make_result(tmp_path / 's1--MID00001-a.npz', protocol='TRA_SWI--nm-pt')
    make_result(tmp_path / 's1--MID00002-b.npz', protocol='t1_mprage--nm-pt')
    windows, scans = ev.evaluate([str(tmp_path)], params={'minWindowS': 3})
    summary = {r['group']: r for r in ev.summarize(windows, scans)}
    assert summary['MPRAGE']['context'] and summary['pooled']['scans'] == 1

# ----- Command line ------------------------------------------------------------------------------

def test_command_line_with_sweep_and_plots(tmp_path):
    pytest.importorskip('pandas')
    data = tmp_path / 'data'; data.mkdir()
    make_result(data / 's1--MID00001-still.npz', protocol='TRA_SWI--nm-pt')
    make_result(data / 's1--MID00002-move.npz', protocol='TRA_SWI--m-pt', moveAtS=31.0)
    out = tmp_path / 'out'
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, os.path.join(repoDir, 'ptone', 'bulk_motion_eval.py'), str(data),
                             '--out-dir', str(out), '--sweep', 'minWindowS=2,3', '--plots'],
                            capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stderr
    assert '[minWindowS=2]' in result.stdout and '[minWindowS=3]' in result.stdout
    import pandas
    summary = pandas.read_csv(out / 'summary.csv')
    assert sorted(set(summary['params'])) == ['{"minWindowS": "2"}', '{"minWindowS": "3"}']
    windows = pandas.read_csv(out / 'windows.csv')
    assert {'score', 'trackerRmsMm', 'trackerMm', 'trackerDeg', 'moved', 'still', 'windowStartS'} <= set(windows.columns)
    assert (out / 'sessions.csv').exists()
    assert len(os.listdir(out / 'plots--minWindowS-2')) == 2

def test_command_line_no_tcl_data(tmp_path):
    result = subprocess.run([sys.executable, os.path.join(repoDir, 'ptone', 'bulk_motion_eval.py'), str(tmp_path),
                             '--out-dir', str(tmp_path / 'out')], capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)), timeout=300)
    assert result.returncode == 1 and 'no scores' in result.stderr
