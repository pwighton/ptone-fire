# Tests for ptone/tone_plot.py: plotting k-space lines with the pilot tone fitted and subtracted
#
# The twix test looks for ptone20260114 in $PTONE_TEST_DATA (default ../pilot-tone-test-data relative to the
# repository root), and is skipped if it's not found or twixtools isn't installed.

import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from ptone.tone_plot import plot_lines, read_lines
from ptone.tests.test_tone_estimation import make_line

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataDir = os.environ.get('PTONE_TEST_DATA', os.path.join(repoDir, '..', 'pilot-tone-test-data'))

F0_HZ = 123249206.0
DWELL_US = 9.8
TONE_CYCLES = 0.3729            # Tone frequency, cycles per sample
TX_HZ = F0_HZ + TONE_CYCLES / (DWELL_US * 1e-6)

HEADER = '''<?xml version="1.0" encoding="utf-8"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
<measurementInformation><patientPosition>HFS</patientPosition><protocolName>test_tse--nm-pt</protocolName></measurementInformation>
<experimentalConditions><H1resonanceFrequency_Hz>%d</H1resonanceFrequency_Hz></experimentalConditions>
<encoding>
<encodedSpace><matrixSize><x>512</x><y>8</y><z>1</z></matrixSize><fieldOfView_mm><x>480</x><y>240</y><z>5</z></fieldOfView_mm></encodedSpace>
<reconSpace><matrixSize><x>256</x><y>8</y><z>1</z></matrixSize><fieldOfView_mm><x>240</x><y>240</y><z>5</z></fieldOfView_mm></reconSpace>
<encodingLimits></encodingLimits>
<trajectory>cartesian</trajectory>
</encoding>
</ismrmrdHeader>''' % F0_HZ

def make_mrd(path, numLines=6, firstScanCounter=10):
    # An MRD file of lines with a tone at TONE_CYCLES (readout through the isocentre, so its nominal frequency is
    # TX_HZ - F0_HZ).  Returns {scanCounter: line without the tone}
    import ismrmrd
    ds = ismrmrd.Dataset(str(path), 'dataset', create_if_needed=True)
    clean = {}
    try:
        ds.write_xml_header(HEADER)
        for i in range(numLines):
            line, withoutTone, _ = make_line(TONE_CYCLES, numChan=8, seed=i)
            acq = ismrmrd.Acquisition.from_array(line)
            acq.scan_counter = firstScanCounter + i
            acq.sample_time_us = DWELL_US
            acq.read_dir[:] = (1, 0, 0)
            ds.append_acquisition(acq)
            clean[acq.scan_counter] = withoutTone
    finally:
        ds.close()
    return clean

def png_size(path):
    with open(path, 'rb') as f:
        data = f.read(24)
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', data[16:24])

def test_read_mrd_lines(tmp_path):
    path = tmp_path / 'scan.mrd'
    make_mrd(path)
    info, lines = read_lines(str(path), [12, 10])
    assert info['protocol'] == 'test_tse--nm-pt' and info['f0Hz'] == F0_HZ
    assert info['imagingHalfBand'] == pytest.approx(0.25)          # 2x oversampling
    assert sorted(lines) == [10, 12] and lines[10]['data'].shape == (8, 512)
    assert lines[10]['dwellS'] == pytest.approx(DWELL_US * 1e-6)
    assert lines[10]['nominal'](TX_HZ) == pytest.approx(TONE_CYCLES / (DWELL_US * 1e-6))
    with pytest.raises(ValueError, match='Line\\(s\\) 99 not found'):
        read_lines(str(path), [10, 99])

def test_plot_lines(tmp_path):
    path = tmp_path / 'scan.mrd'
    make_mrd(path)
    png, fits = plot_lines(str(path), [10, 13], channels=[0, 5], zoomBins=10, txFreqHz=TX_HZ)
    assert png == str(tmp_path / 'scan--tone--lines-10-13.png')
    width, height = png_size(png)
    assert width == 2 * 2100 and height > 2 * 4 * 300             # 3 panels of 7 in at 200 dpi; 4 rows (2 lines x 2 channels)
    assert set(fits) == {10, 13}
    # Within 0.005 of a bin: with 8 channels, one line pins the frequency down less well than with more
    assert fits[10]['freq'] == pytest.approx(TONE_CYCLES, abs=5e-3 / 512)
    assert fits[10]['freqHz'] == pytest.approx(TONE_CYCLES / (DWELL_US * 1e-6), abs=1.0)
    assert fits[10]['peakToNoiseDb'] > 30
    png2, _ = plot_lines(str(path), [11], channels=[3], pngPath=str(tmp_path / 'one.png'), part='abs')
    assert png_size(png2)[0] == 2 * 1400                           # 2 panels without the zoom
    png3, _ = plot_lines(str(path), [11], channels=[3], pngPath=str(tmp_path / 'low.png'), dpi=100)
    assert png_size(png3)[0] == 1400

def test_plot_errors(tmp_path):
    path = tmp_path / 'scan.mrd'
    make_mrd(path)
    with pytest.raises(ValueError, match='Channel'):
        plot_lines(str(path), [10], channels=[8])
    with pytest.raises(ValueError, match='part'):
        plot_lines(str(path), [10], part='phase')

def test_command_line(tmp_path):
    path = tmp_path / 'scan.mrd'
    make_mrd(path)
    out = tmp_path / 'x.png'
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, os.path.join(repoDir, 'ptone', 'tone_plot.py'), str(path), '--lines', '11',
                             '--channels', '0', '2', '--out', str(out), '--tx-freq', str(TX_HZ)],
                            capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'line 11: tone at 0.3729' in result.stdout and out.exists()
    result = subprocess.run([sys.executable, os.path.join(repoDir, 'ptone', 'tone_plot.py'), str(path), '--lines', '99'],
                            capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 1 and 'not found' in result.stderr

TWIX_FILES = [
    # (file, tone expected, transmit frequency from the session's scripts)
    ('meas_MID00900_FID47882_t2_tse_tra_dark_fluid_ARIA__nm_pt.dat', True, 123.285661e6),
    ('meas_MID00902_FID47884_t2_tse_tra_dark_fluid_ARIA__nm_npt.dat', False, None),
]

@pytest.mark.parametrize('fileName, hasTone, txHz', TWIX_FILES)
def test_twix_real_data(tmp_path, fileName, hasTone, txHz):
    pytest.importorskip('twixtools')
    path = os.path.join(dataDir, 'ptone20260114', fileName)
    if not os.path.exists(path):
        pytest.skip('Not found: ' + path)
    png, fits = plot_lines(path, [1500], channels=[5], pngPath=str(tmp_path / 'twix.png'), txFreqHz=txHz)
    assert png_size(png)[0] == 2 * 1400
    if hasTone:
        assert fits[1500]['peakToNoiseDb'] > 30
        # The fitted tone is within a bin (here ~200 Hz) of where the transmit frequency puts it
        info, lines = read_lines(path, [1500])
        assert abs(fits[1500]['freqHz'] - lines[1500]['nominal'](txHz)) < 1 / (512 * 9.8e-6)
    else:
        assert fits[1500]['peakToNoiseDb'] < 15
