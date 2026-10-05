# Tests for ptone/ptone_plot.py, and pilottone.py's automatic plot

import json
import os
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from ptone.ptone_plot import plot_npz, short_git_commit
from ptone.tests.test_tcl import write_tcl

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plotScript = os.path.join(repoDir, 'ptone', 'ptone_plot.py')

def make_npz(path, numLines=50, numChan=4, firstLine=10, lastScanCounter=200, startMs=42578541.0, withLast=True):
    # A results file with the same contents pilottone.py saves
    rng = np.random.default_rng(0)
    settings = {'refChanIdx': 0, 'ptoneTx': True, 'ptoneTxFreqHz': 123290943.0, 'ptoneTxDB': 70,
                'ptoneTxStartScanCounter': firstLine - 2, 'gitCommit': 'abc123'}
    amplitude = 1 + 0.1 * rng.random((numLines, numChan))
    phase = rng.random((numLines, numChan))
    contents = dict(
        scan_counter=np.arange(firstLine, firstLine + numLines), line=np.arange(numLines), slice=np.zeros(numLines),
        time_ms=startMs + 100 * np.arange(numLines), amplitude=amplitude, phase=phase,
        quality=0.9 + 0.05 * rng.random(numLines),
        relative_amplitude=amplitude / amplitude[:, :1], relative_phase=phase - phase[:, :1],
        timestamp=np.array('20261001-120000-000'), config=np.array('{}'), settings=np.array(json.dumps(settings)),
        mrd_header=np.array('<ismrmrdHeader><measurementInformation><protocolName>test_protocol'
                            '</protocolName></measurementInformation></ismrmrdHeader>'))
    if withLast:
        contents['last_scan_counter'] = np.array(lastScanCounter)
    np.savez(path, **contents)
    return str(path)

def png_size(path):
    # (width, height) from the PNG header
    with open(path, 'rb') as f:
        data = f.read(24)
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', data[16:24])

def run_cli(*args):
    # Run ptone_plot.py as a script, with this Python's module path (e.g. for pandas)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    return subprocess.run([sys.executable, plotScript] + [str(a) for a in args], capture_output=True, text=True,
                          env=env, timeout=120)

def test_default_png_path(tmp_path):
    npz = make_npz(tmp_path / 'results.npz')
    assert plot_npz(npz) == str(tmp_path / 'results.png')
    png_size(tmp_path / 'results.png')

def test_height_scales_with_channels(tmp_path):
    npz = make_npz(tmp_path / 'results.npz', numChan=4)
    _, h1 = png_size(plot_npz(npz, pngPath=str(tmp_path / 'one.png'), channels=[0]))
    _, h4 = png_size(plot_npz(npz, pngPath=str(tmp_path / 'four.png')))
    assert h4 > 3 * h1

def test_cli_with_options(tmp_path):
    npz = make_npz(tmp_path / 'results.npz')
    out = tmp_path / 'x.png'
    result = run_cli(npz, '--channel', 0, 1, '--plot', 'phase', '--amplitude', 'raw', '--no-quality',
                     '--quality-threshold', 0.92, '--legend-loc', 'none', '--outfile', out)
    assert result.returncode == 0, result.stderr
    assert 'Saved plot to' in result.stdout
    png_size(out)

def test_bad_channel(tmp_path):
    npz = make_npz(tmp_path / 'results.npz', numChan=4)
    with pytest.raises(ValueError, match='out of range'):
        plot_npz(npz, channels=[4])
    result = run_cli(npz, '--channel', 4)
    assert result.returncode == 1 and 'out of range' in result.stderr

@pytest.mark.parametrize('full, short', [
    ('35c5d11d533e4dcccc50cbde79bd8306675a4088', '35c5d11'),
    ('35c5d11d533e4dcccc50cbde79bd8306675a4088 (with uncommitted changes)', '35c5d11 (with uncommitted changes)'),
    ("unknown ([Errno 2] No such file or directory: 'git')", "unknown ([Errno 2] No such file or directory: 'git')"),
])
def test_short_git_commit(full, short):
    assert short_git_commit(full) == short

def test_older_file_without_last_scan_counter(tmp_path):
    npz = make_npz(tmp_path / 'results.npz', withLast=False)
    png_size(plot_npz(npz))

def test_tcl_motion(tmp_path):
    pytest.importorskip('pandas')
    motPath = str(tmp_path / 'session_120616_MOT.tsm')
    startMs = write_tcl(motPath, numSamples=100)
    npz = make_npz(tmp_path / 'results.npz', startMs=startMs)
    png_size(plot_npz(npz, pngPath=str(tmp_path / 'python.png'), tclMotPath=motPath))
    result = run_cli(npz, '--tcl', motPath, '--outfile', tmp_path / 'cli.png')
    assert result.returncode == 0, result.stderr
    png_size(tmp_path / 'cli.png')

def test_tcl_motion_from_npz(tmp_path):
    # TCL data stored in the results file is plotted exactly as if it was given with --tcl
    pytest.importorskip('pandas')
    from ptone.tcl import add_tcl_to_npz
    motPath = str(tmp_path / 'session_120616_MOT.tsm')
    startMs = write_tcl(motPath, numSamples=100)
    plain = make_npz(tmp_path / 'plain.npz', startMs=startMs)
    withTcl = make_npz(tmp_path / 'with_tcl.npz', startMs=startMs)
    add_tcl_to_npz(withTcl, motPath)
    fromNpz = plot_npz(withTcl, pngPath=str(tmp_path / 'from_npz.png'))
    fromFile = plot_npz(plain, pngPath=str(tmp_path / 'from_file.png'), tclMotPath=motPath)
    noMotion = plot_npz(plain, pngPath=str(tmp_path / 'no_motion.png'))
    read = lambda p: open(p, 'rb').read()
    assert read(fromNpz) == read(fromFile)
    assert read(fromNpz) != read(noMotion)

def test_add_tcl(tmp_path):
    # --add-tcl saves the motion into the results file; its plot and later plots without --tcl match a
    # plot made with --tcl
    pytest.importorskip('pandas')
    motPath = str(tmp_path / 'session_120616_MOT.tsm')
    startMs = write_tcl(motPath, numSamples=100)
    plain = make_npz(tmp_path / 'plain.npz', startMs=startMs)
    viaCli = make_npz(tmp_path / 'cli.npz', startMs=startMs)
    viaPython = make_npz(tmp_path / 'python.npz', startMs=startMs)
    reference = open(plot_npz(plain, pngPath=str(tmp_path / 'reference.png'), tclMotPath=motPath), 'rb').read()

    result = run_cli(viaCli, '--tcl', motPath, '--add-tcl')
    assert result.returncode == 0, result.stderr
    assert 'Added TCL motion' in result.stdout
    assert open(viaCli.replace('.npz', '.png'), 'rb').read() == reference
    assert str(np.load(viaCli)['tcl_file']) == os.path.abspath(motPath)

    plot_npz(viaPython, tclMotPath=motPath, addTcl=True)
    assert 'tcl_3d_motion' in np.load(viaPython).files
    # Plotted again later without --tcl: the saved motion is shown
    assert open(plot_npz(viaPython, pngPath=str(tmp_path / 'later.png')), 'rb').read() == reference

def test_add_tcl_errors(tmp_path):
    pytest.importorskip('pandas')
    motPath = str(tmp_path / 'session_120616_MOT.tsm')
    startMs = write_tcl(motPath, numSamples=20)                                     # 2 s of motion data
    npz = make_npz(tmp_path / 'results.npz', startMs=startMs)                        # 5 s of lines
    before = open(npz, 'rb').read()
    result = run_cli(npz, '--add-tcl')
    assert result.returncode == 1 and '--add-tcl needs a TCL motion file' in result.stderr
    result = run_cli(npz, '--tcl', motPath, '--add-tcl')
    assert result.returncode == 1 and "doesn't cover the whole scan" in result.stderr
    assert open(npz, 'rb').read() == before
    with pytest.raises(ValueError):
        plot_npz(npz, addTcl=True)

def test_tcl_missing(tmp_path):
    npz = make_npz(tmp_path / 'results.npz')
    result = run_cli(npz, '--tcl', tmp_path / 'nonexistent_MOT.tsm')
    assert result.returncode == 1 and 'TCL motion file not found' in result.stderr

# ----- pilottone.py's automatic plot ---------------------------------------------------------

@pytest.mark.parametrize('ptonePlot', ['true', 'false'])
def test_pilottone_plots_after_scan(tmp_path, ptonePlot):
    import ismrmrd
    sys.path.insert(0, repoDir)
    import pilottone

    class FakeConnection(list):
        def send_close(self):
            pass

    acqs = FakeConnection()
    for i in range(20):
        data = np.exp(2j * np.pi * 0.3 * np.arange(256))[None, :].repeat(4, 0) + 0.01 * np.random.randn(4, 256)
        acq = ismrmrd.Acquisition.from_array(data.astype(np.complex64))
        acq.scan_counter = i + 1
        acq.acquisition_time_stamp = 4 * i
        acqs.append(acq)

    pilottone.process(acqs, {'parameters': {'outputFolder': str(tmp_path), 'ptonePlot': ptonePlot}}, None)
    npz = [f for f in os.listdir(tmp_path) if f.endswith('.npz')][0]
    assert int(np.load(tmp_path / npz)['last_scan_counter']) == 20
    png = tmp_path / npz.replace('.npz', '.png')
    log = tmp_path / npz.replace('.npz', '.txt')

    if ptonePlot == 'true':
        # Plotted in a separate process, so wait for it
        end = time.monotonic() + 60
        while 'Saved plot to' not in log.read_text() and time.monotonic() < end:
            time.sleep(0.2)
        assert 'Saved plot to' in log.read_text()
        png_size(png)
    else:
        time.sleep(2)
        assert not png.exists()
