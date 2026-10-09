# Tests for pilottone.py's output naming
#
# The real-header tests look for data in $PTONE_TEST_DATA (default ../pilot-tone-test-data relative to the
# repository root), including its 20260930-bay1-tests folder, and are skipped if not found.

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
    (os.path.join(dataDir, '20260930-bay1-tests', 'bay1-test1.h5'), 1104),   # Header from the FIRE stream
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

def run_flagged(tmp_path, delayMs, numFlagged, numLines=6, flag='ACQ_IS_NAVIGATION_DATA', skipFlags='ACQ_IS_NOISE_MEASUREMENT'):
    # Lines whose first numFlagged carry flag.  With the defaults: ACQ_IS_NAVIGATION_DATA, which skipFlags (here
    # only the noise flag) doesn't skip but ptoneTxFreqSkipFlags (default) does, so those lines are analysed but
    # can't start the transmitter.  skipFlags None: the default skipFlags.  Returns the scan_counters analysed
    # (empty if no results were saved)
    import ismrmrd
    acqs = FakeConnection()
    for i in range(numLines):
        data = np.exp(2j * np.pi * 0.3 * np.arange(256))[None, :].repeat(4, 0) + 0.01 * np.random.randn(4, 256)
        acq = ismrmrd.Acquisition.from_array(data.astype(np.complex64))
        acq.scan_counter = i + 1
        acq.acquisition_time_stamp = 4 * i
        if i < numFlagged:
            acq.set_flag(getattr(ismrmrd, flag))
        acqs.append(acq)
    params = {'outputFolder': str(tmp_path), 'ptonePlot': 'false', 'ptoneTxDelayMs': str(delayMs)}
    if skipFlags is not None:
        params['skipFlags'] = skipFlags
    pilottone.process(acqs, {'parameters': params}, None)
    npz = [f for f in os.listdir(tmp_path) if f.endswith('.npz')]
    return list(np.load(tmp_path / npz[0])['scan_counter']) if npz else []

def test_all_lines_flagged_needs_negative_delay(tmp_path):
    # Like the MPRAGE: no line can start the transmitter, so with delay 0 nothing is analysed ...
    assert run_flagged(tmp_path / 'zero', 0, numFlagged=6) == []
    # ... and with a negative delay every eligible line is
    assert run_flagged(tmp_path / 'negative', -1, numFlagged=6) == [1, 2, 3, 4, 5, 6]

def test_coil_correction_flag_doesnt_skip_lines(tmp_path):
    # ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA is the Siemens RAWDATACORRECTION flag, which the scanner sets on every
    # line of e.g. SWI and MPRAGE scans: with the default flag lists those lines start the transmitter and are
    # analysed
    assert run_flagged(tmp_path, 0, numFlagged=6, flag='ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA', skipFlags=None) == [1, 2, 3, 4, 5, 6]
    assert 'ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA' not in pilottone.defaultSkipFlags
    assert 'ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA' not in pilottone.defaultPtoneTxFreqSkipFlags

def test_warning_when_transmitter_never_starts(tmp_path, caplog):
    # Every line has a ptoneTxFreqSkipFlags flag: a warning, which says nothing was analysed if ptoneTxDelayMs >= 0
    assert run_flagged(tmp_path / 'zero', 0, numFlagged=6) == []
    assert "No line could set the pilot tone frequency or start the transmitter" in caplog.text
    assert "no lines were analysed" in caplog.text
    caplog.clear()
    assert run_flagged(tmp_path / 'negative', -1, numFlagged=6) == [1, 2, 3, 4, 5, 6]
    assert "No line could set the pilot tone frequency" in caplog.text and "no lines were analysed" not in caplog.text
    caplog.clear()
    # One line can start it: no warning
    run_flagged(tmp_path / 'one', 0, numFlagged=5)
    assert "No line could set the pilot tone frequency" not in caplog.text

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

# ----- Bulk motion score -------------------------------------------------------------------------

def run_moving_tone(tmp_path, params=None, numNoise=10, durationS=14.0, stepAtS=7.0, stepRad=0.2, header=None):
    # numNoise lines without the tone, then durationS of lines (one per 40 ms) with the tone, whose phase on
    # channels 1-3 steps by stepRad relative to channel 0 stepAtS after the tone appears (the head moving).
    # Returns the saved results, their settings, and the results file's path (None if nothing was saved)
    import ismrmrd, json
    rng = np.random.default_rng(5)
    acqs = FakeConnection()
    n = np.arange(256)
    numTone = int(durationS * 1000 / 40)
    for i in range(numNoise + numTone):
        noise = 0.01 * (rng.standard_normal((4, 256)) + 1j * rng.standard_normal((4, 256)))
        if i < numNoise:
            data = noise
        else:
            step = stepRad if (i - numNoise) * 0.04 >= stepAtS else 0.0
            phases = np.array([0.0, 1.0 + step, 3.0 + step, -2.0 - step])
            data = np.exp(1j * phases)[:, None] * np.exp(2j * np.pi * 0.3 * n)[None, :] + noise
        acq = ismrmrd.Acquisition.from_array(data.astype(np.complex64))
        acq.scan_counter = i + 1
        acq.acquisition_time_stamp = 16 * i            # 40 ms apart (2.5 ms ticks)
        acqs.append(acq)
    config = {'outputFolder': str(tmp_path), 'ptonePlot': 'false', 'ptoneTxDelayMs': '-1',
              'skipFlags': 'ACQ_IS_NOISE_MEASUREMENT'}
    config.update(params or {})
    pilottone.process(acqs, {'parameters': config}, header)
    npz = [f for f in os.listdir(tmp_path) if f.endswith('.npz')]
    if not npz:
        return None, None, None
    d = np.load(tmp_path / npz[0])
    return d, json.loads(str(d['settings'])), str(tmp_path / npz[0])

def replay(d, settings, **params):
    # The scores medianFilter gives on the saved results, line by line (as bulk_motion_eval.py does)
    from ptone.bulk_motion import MedianFilter
    m = MedianFilter(None, settings['refChanIdx'], minQuality=settings['ptoneQualityThreshold'], **params)
    scores = [m.update(t, ph, amp, q) for t, ph, amp, q in
              zip(d['time_ms'], d['relative_phase'], d['relative_amplitude'], d['quality'])]
    return np.array([np.nan if s is None else s for s in scores])

def test_bulk_motion_score_default(tmp_path):
    d, settings, _ = run_moving_tone(tmp_path)
    assert settings['bulkMotionMethod'] == 'medianFilter'
    assert settings['medianFilterWindowS'] == 3.0 and settings['medianFilterMinLinesPerWindow'] == 5
    assert settings['medianFilterMaxGapS'] is None and settings['bulkMotionError'] is None
    score = d['bulk_motion_score']
    assert score.shape == d['quality'].shape
    # Windows of 3 s from the first tone line (line 11): 0-3, 3-6, 6-9, 9-12 s, then 12-14 s never completes.
    # Scores for the 3-6, 6-9 and 9-12 s windows, on the lines that complete them
    hasScore = np.flatnonzero(np.isfinite(score))
    assert list(d['scan_counter'][hasScore]) == [11 + 75 * k for k in (2, 3, 4)]
    assert settings['bulkMotionNumScores'] == 3
    np.testing.assert_array_equal(score, replay(d, settings, windowS=3))
    # The step at 7 s is in the 6-9 s window (more than half of it after the step): RMS over channels 1-3
    # of 0.2 rad; the other windows don't change
    assert score[hasScore[1]] == pytest.approx(0.2, abs=0.01)
    assert score[hasScore[0]] < 0.01 and score[hasScore[2]] < 0.01

def test_bulk_motion_parameters_from_config(tmp_path, caplog):
    caplog.set_level('INFO')
    d, settings, path = run_moving_tone(tmp_path, {'medianFilterWindowS': '2', 'medianFilterMinLinesPerWindow': '10',
                                                   'medianFilterMaxGapS': '', 'ptoneQualityThreshold': '0.6'})
    assert settings['medianFilterWindowS'] == 2.0 and settings['medianFilterMinLinesPerWindow'] == 10
    assert settings['medianFilterMaxGapS'] is None
    assert settings['bulkMotionNumScores'] == 5                     # 2 s windows: 0-2, ..., 10-12 s; 12-14 s incomplete
    np.testing.assert_array_equal(d['bulk_motion_score'], replay(d, settings, windowS=2, minLinesPerWindow=10))
    log = caplog.text
    assert 'Bulk motion score: medianFilter, windowS 2.0, minLinesPerWindow 10, maxGapS None' in log
    assert log.count('Bulk motion score 0.') == 5

def test_bulk_motion_tr_from_header(tmp_path, caplog):
    caplog.set_level('INFO')
    header = SimpleNamespace(measurementInformation=SimpleNamespace(measurementID='1_2_3', protocolName='p'),
                             sequenceParameters=SimpleNamespace(TR=[597.0]))
    assert pilottone.get_tr_ms(header) == 597.0
    assert pilottone.get_tr_ms(None) is None
    assert pilottone.get_tr_ms(SimpleNamespace(sequenceParameters=SimpleNamespace(TR=[]))) is None
    d, settings, path = run_moving_tone(tmp_path, header=header)
    assert '(TR 597.0 ms' in caplog.text

def test_bulk_motion_none(tmp_path):
    d, settings, _ = run_moving_tone(tmp_path, {'bulkMotionMethod': 'none', 'medianFilterWindowS': '2'})
    assert settings['bulkMotionMethod'] == 'none'
    assert 'bulk_motion_score' not in d.files and 'medianFilterWindowS' not in settings

@pytest.mark.parametrize('params, message', [
    ({'bulkMotionMethod': 'nonsense'}, "Unknown bulkMotionMethod 'nonsense'"),
    ({'medianFilterWindowS': '0'}, 'windowS must be positive'),
])
def test_bulk_motion_bad_settings(tmp_path, params, message):
    # Like a bad flag name: the error is logged and nothing is analysed
    d, _, _ = run_moving_tone(tmp_path, params)
    assert d is None
    log = [f for f in os.listdir(tmp_path) if f.endswith('.txt')][0]
    assert message in open(tmp_path / log).read()

def test_get_bulk_motion_params():
    config = {'parameters': {'medianFilterWindowS': '2', 'medianFilterMaxGapS': ' ', 'otherWindowS': '9'}}
    assert pilottone.get_bulk_motion_params(config, 'medianFilter') == {'windowS': '2'}
    assert pilottone.get_bulk_motion_params(None, 'medianFilter') == {}
    assert pilottone.get_bulk_motion_params(config, 'none') is None
    with pytest.raises(ValueError, match='Unknown bulkMotionMethod'):
        pilottone.get_bulk_motion_params(config, 'other')

# ----- Settings by protocol name and sequence type (protocolOverrides) ---------------------------

def header_with_protocol(protocolName, sequenceType=None):
    return SimpleNamespace(measurementInformation=SimpleNamespace(measurementID='1_2_7', protocolName=protocolName),
                           sequenceParameters=SimpleNamespace(TR=[9000.0], sequence_type=sequenceType))

OVERRIDES = [{'matchProtocolName': '*tse*', 'medianFilterWindowS': '2'},
             {'matchProtocolName': '*swi*', 'medianFilterWindowS': '5'},
             {'matchSequenceType': 'TurboSpinEcho', 'medianFilterMinLinesPerWindow': '10'}]

def test_protocol_overrides_applied(tmp_path, caplog):
    caplog.set_level('INFO')
    d, settings, _ = run_moving_tone(tmp_path, {'protocolOverrides': OVERRIDES},
                                     header=header_with_protocol('t2_TSE_tra--m-pt', 'TurboSpinEcho'))
    # Both the protocol name rule and the sequence type rule apply
    assert settings['medianFilterWindowS'] == 2.0 and settings['medianFilterMinLinesPerWindow'] == 10
    assert settings['bulkMotionNumScores'] == 5                     # 2 s windows, as in test_bulk_motion_parameters_from_config
    assert settings['protocolName'] == 't2_TSE_tra--m-pt' and settings['sequenceType'] == 'TurboSpinEcho'
    assert settings['protocolOverrideIndices'] == [0, 2]
    assert settings['protocolOverrideSettings'] == {'medianFilterWindowS': '2', 'medianFilterMinLinesPerWindow': '10'}
    import json
    assert json.loads(str(d['config']))['parameters']['protocolOverrides'] == OVERRIDES   # Saved as received
    label = "Protocol 't2_TSE_tra--m-pt', sequence type 'TurboSpinEcho' matches protocolOverrides "
    assert label + "rule 0 (matchProtocolName '*tse*'), which sets medianFilterWindowS = 2" in caplog.text
    assert label + "rule 2 (matchSequenceType 'TurboSpinEcho'), which sets medianFilterMinLinesPerWindow = 10" in caplog.text
    assert "Settings from protocolOverrides: medianFilterWindowS = 2, medianFilterMinLinesPerWindow = 10" in caplog.text

def test_protocol_overrides_no_match(tmp_path, caplog):
    caplog.set_level('INFO')
    d, settings, _ = run_moving_tone(tmp_path, {'protocolOverrides': OVERRIDES, 'medianFilterWindowS': '4'},
                                     header=header_with_protocol('t2_fl2d_tra_hemo--nm-pt', 'Flash'))
    assert settings['medianFilterWindowS'] == 4.0
    assert settings['protocolOverrideIndices'] == [] and settings['protocolOverrideSettings'] is None
    assert "Protocol 't2_fl2d_tra_hemo--nm-pt', sequence type 'Flash' matches no protocolOverrides rule" in caplog.text

def test_protocol_overrides_any_setting(tmp_path):
    # A rule can set any setting, e.g. turn the bulk motion score off
    rules = [{'matchSequenceType': 'Flash', 'matchProtocolName': '*mprage*', 'bulkMotionMethod': 'none'}]
    d, settings, _ = run_moving_tone(tmp_path, {'protocolOverrides': rules}, header=header_with_protocol('t1_mprage--nm-pt', 'Flash'))
    assert settings['bulkMotionMethod'] == 'none' and 'bulk_motion_score' not in d.files

def test_protocol_overrides_malformed(tmp_path):
    # Like a bad flag name: the error is logged and nothing is analysed
    d, _, _ = run_moving_tone(tmp_path, {'protocolOverrides': [{'match': '*tse*', 'medianFilterWindowS': '2'}]},
                              header=header_with_protocol('t2_tse'))
    assert d is None
    log = [f for f in os.listdir(tmp_path) if f.endswith('.txt')][0]
    assert "now 'matchProtocolName'" in open(tmp_path / log).read()

def test_header_sequence_type():
    assert pilottone.get_header_sequence_type(header_with_protocol('x', 'Flash')) == 'Flash'
    assert pilottone.get_header_sequence_type(header_with_protocol('x')) is None
    assert pilottone.get_header_sequence_type(None) is None
    assert pilottone.get_header_sequence_type('not valid MRD XML') is None
