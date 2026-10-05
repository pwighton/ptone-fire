# Tests for pilottone.py's output naming
#
# The real-header tests look for data in $PTONE_TEST_DATA (default ../pilot-tone-test-data) and
# ../20260930-bay1-tests relative to the repository root, and are skipped if not found.

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repoDir)
import pilottone

dataDir = os.environ.get('PTONE_TEST_DATA', os.path.join(repoDir, '..', 'pilot-tone-test-data'))

def header_with_measurement_id(measurementID):
    return SimpleNamespace(measurementInformation=SimpleNamespace(measurementID=measurementID, protocolName='test_protocol'))

@pytest.mark.parametrize('measurementID, mid', [
    ('45407_0000026092514332225800000049_0000026092514332225800000049_1104', 1104),
    ('67026_0000026042812130895600000021_0000026042812130895600000021_284', 284),
    ('284', 284),
    ('45407_abc_def', None),
    ('', None),
    (None, None),
])
def test_get_mid(measurementID, mid):
    assert pilottone.get_mid(header_with_measurement_id(measurementID)) == mid

@pytest.mark.parametrize('header', [None, 'not valid MRD XML', SimpleNamespace()])
def test_get_mid_without_header(header):
    assert pilottone.get_mid(header) is None

@pytest.mark.parametrize('path, mid', [
    (os.path.join(dataDir, 'ptoneH20260429--meas_MID00284_FID14131_t2_tse_tra_dark_fluid__m_pt.h5'), 284),
    (os.path.join(repoDir, '..', '20260930-bay1-tests', 'bay1-test1.h5'), 1104),   # Header from the FIRE stream
])
def test_get_mid_real_headers(path, mid):
    if not os.path.exists(path):
        pytest.skip('Not found: ' + path)
    import h5py, ismrmrd
    with h5py.File(path, 'r') as f:
        header = ismrmrd.xsd.CreateFromDocument(f['dataset']['xml'][0])
    assert pilottone.get_mid(header) == mid

class FakeConnection(list):
    def send_close(self):
        pass

def run_process(tmp_path, header):
    import ismrmrd
    acqs = FakeConnection()
    for i in range(5):
        data = np.exp(2j * np.pi * 0.3 * np.arange(256))[None, :].repeat(4, 0) + 0.01 * np.random.randn(4, 256)
        acq = ismrmrd.Acquisition.from_array(data.astype(np.complex64))
        acq.scan_counter = i + 1
        acq.acquisition_time_stamp = 4 * i
        acqs.append(acq)
    pilottone.process(acqs, {'parameters': {'outputFolder': str(tmp_path), 'outputFileStem': 'sub01', 'ptonePlot': 'false'}}, header)
    return sorted(os.listdir(tmp_path))

def test_output_filename_has_mid(tmp_path):
    files = run_process(tmp_path, header_with_measurement_id('45407_x_y_1104'))
    assert [f.rsplit('--', 1)[0] for f in files] == ['sub01--MID01104-test_protocol'] * 2
    assert {os.path.splitext(f)[1] for f in files} == {'.npz', '.txt'}

def test_output_filename_without_mid(tmp_path):
    files = run_process(tmp_path, None)
    assert all(f.startswith('sub01--MIDunknown-unknown--') for f in files)

# ----- Negative ptoneTxDelayMs: analyse every eligible line ----------------------------------------

def run_flagged(tmp_path, delayMs, numFlagged, numLines=6):
    # Lines whose first numFlagged carry ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA (as siemens_to_ismrmrd does for
    # every MPRAGE line).  skipFlags doesn't skip that flag; ptoneTxFreqSkipFlags (default) does, so those lines
    # can't start the transmitter.  Returns the scan_counters analysed (empty if no results were saved)
    import ismrmrd
    acqs = FakeConnection()
    for i in range(numLines):
        data = np.exp(2j * np.pi * 0.3 * np.arange(256))[None, :].repeat(4, 0) + 0.01 * np.random.randn(4, 256)
        acq = ismrmrd.Acquisition.from_array(data.astype(np.complex64))
        acq.scan_counter = i + 1
        acq.acquisition_time_stamp = 4 * i
        if i < numFlagged:
            acq.set_flag(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA)
        acqs.append(acq)
    pilottone.process(acqs, {'parameters': {'outputFolder': str(tmp_path), 'ptonePlot': 'false', 'ptoneTxDelayMs': str(delayMs),
                                            'skipFlags': 'ACQ_IS_NOISE_MEASUREMENT'}}, None)
    npz = [f for f in os.listdir(tmp_path) if f.endswith('.npz')]
    return list(np.load(tmp_path / npz[0])['scan_counter']) if npz else []

def test_all_lines_flagged_needs_negative_delay(tmp_path):
    # Like the MPRAGE: no line can start the transmitter, so with delay 0 nothing is analysed ...
    assert run_flagged(tmp_path / 'zero', 0, numFlagged=6) == []
    # ... and with a negative delay every eligible line is
    assert run_flagged(tmp_path / 'negative', -1, numFlagged=6) == [1, 2, 3, 4, 5, 6]

def test_negative_delay_includes_lines_before_transmitter_start(tmp_path):
    # The first 2 lines can't start the transmitter; line 3 does
    assert run_flagged(tmp_path / 'zero', 0, numFlagged=2) == [3, 4, 5, 6]
    assert run_flagged(tmp_path / 'negative', -1, numFlagged=2) == [1, 2, 3, 4, 5, 6]

# ----- Phase range midpoint ----------------------------------------------------------------------

def run_noise_then_tone(tmp_path, numNoise=4, numTone=20, threshold=None):
    # numNoise lines of noise only (no tone: low quality), then numTone lines with a tone whose phase on
    # channel 2 drifts from 3.0 to 3.5 rad relative to channel 0.  Returns the saved results
    import ismrmrd, json
    rng = np.random.default_rng(3)
    acqs = FakeConnection()
    n = np.arange(256)
    for i in range(numNoise + numTone):
        noise = 0.01 * (rng.standard_normal((4, 256)) + 1j * rng.standard_normal((4, 256)))
        if i < numNoise:
            data = noise
        else:
            phases = np.array([0.0, 1.0, 3.0 + 0.5 * (i - numNoise) / numTone, -2.0])
            data = np.exp(1j * phases)[:, None] * np.exp(2j * np.pi * 0.3 * n)[None, :] + noise
        acq = ismrmrd.Acquisition.from_array(data.astype(np.complex64))
        acq.scan_counter = i + 1
        acq.acquisition_time_stamp = 4 * i
        acqs.append(acq)
    params = {'outputFolder': str(tmp_path), 'ptonePlot': 'false', 'ptoneTxDelayMs': '-1',
              'skipFlags': 'ACQ_IS_NOISE_MEASUREMENT'}
    if threshold is not None:
        params['ptoneQualityThreshold'] = str(threshold)
    pilottone.process(acqs, {'parameters': params}, None)
    npz = [f for f in os.listdir(tmp_path) if f.endswith('.npz')][0]
    d = np.load(tmp_path / npz)
    return d, json.loads(str(d['settings']))

def test_midpoint_from_first_line_with_tone(tmp_path):
    d, settings = run_noise_then_tone(tmp_path)
    q = d['quality']
    assert np.all(q[:4] < 0.5) and np.all(q[4:] > 0.9)          # Noise lines vs tone lines
    assert settings['ptoneQualityThreshold'] == 0.5
    assert settings['phaseMidpointScanCounter'] == 5            # The first tone line
    # Tone lines: relative phase centred on the first tone line's value, continuous (no 2*pi jumps)
    ph = d['relative_phase'][4:, 2]
    assert np.allclose(ph, 3.0 + 0.5 * np.arange(20) / 20, atol=0.05)

def test_midpoint_quality_threshold_zero_uses_first_line(tmp_path):
    # ptoneQualityThreshold 0: the midpoint comes from the first analysed line, even without the tone
    d, settings = run_noise_then_tone(tmp_path, threshold=0)
    assert settings['ptoneQualityThreshold'] == 0
    assert settings['phaseMidpointScanCounter'] == 1
