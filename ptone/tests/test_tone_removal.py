# Tests for ptone/tone_removal.py: subtracting the pilot tone from k-space lines

import numpy as np
import pytest

from ptone.tone_estimation import fit_tone, tone_model
from ptone.tone_removal import ToneRemover, band_power_reduction_db, subtract_tone
from ptone.tests.test_tone_estimation import DWELL, make_line, scan_lines

def test_subtract_tone_leaves_the_rest():
    line, withoutTone, amplitudes = make_line()
    filtered = subtract_tone(line, fit_tone(line))
    assert filtered.dtype == np.complex64 and filtered.shape == line.shape
    # What's left is the line without the tone, apart from the fit's small error (noise-limited)
    residual = np.sqrt(np.mean(np.abs(filtered - withoutTone) ** 2))
    assert residual < 0.02 * np.sqrt(np.mean(np.abs(amplitudes) ** 2))
    assert band_power_reduction_db(line, filtered, fit_tone(line)['freq']) > 30

def test_band_power_reduction():
    line, _, _ = make_line()
    assert band_power_reduction_db(line, line, 0.3787123) == pytest.approx(0.0)
    assert band_power_reduction_db(line, line / 10, 0.3787123) == pytest.approx(20.0)

# ----- ToneRemover: line by line ---------------------------------------------------------------

def test_remover_follows_line_to_line_changes():
    # The frequency jumps between lines by up to a fifth of a bin (like the TSE's per-slice levels) and drifts
    rng = np.random.default_rng(1)
    binHz = 1 / (512 * DWELL)
    freqsHz = 38000 + np.cumsum(rng.uniform(-0.2, 0.2, 40)) * binHz
    lines, clean = scan_lines(freqsHz, [True] * 40)
    remover = ToneRemover()
    for i, line in enumerate(lines):
        res = remover.process(line, DWELL, lineId=i)
        assert res['toneDetected'] and res['searched'] == (i == 0)       # Only the first line needs a search
        assert res['freqHz'] == pytest.approx(freqsHz[i], abs=0.02 * binHz)
        assert res['quality'] > 0.99
        assert np.sqrt(np.mean(np.abs(line - clean[i]) ** 2)) < 0.15     # Tone gone (noise 0.5 per sample)
    assert remover.status()['numToneLines'] == 40 and remover.status()['tonePresent']

def test_lines_without_the_tone_are_untouched():
    # No tone for the first 5 lines (transmitter not started) and lines 12-14 (a gap)
    toneOn = [i >= 5 and not 12 <= i <= 14 for i in range(20)]
    lines, _ = scan_lines(np.full(20, 38000.0), toneOn, seed=2)
    original = [l.copy() for l in lines]
    remover = ToneRemover()
    lastFreq = []
    for i, line in enumerate(lines):
        res = remover.process(line, DWELL, lineId=100 + i)
        assert res['toneDetected'] == toneOn[i]
        if not toneOn[i]:
            np.testing.assert_array_equal(line, original[i])             # Exactly unchanged
            assert res['quality'] < 0.5
        lastFreq.append(remover.lastFreqHz)
    assert lastFreq[11] == lastFreq[12] == lastFreq[14]                  # Lines without the tone aren't warm starts
    st = remover.status()
    assert st['numLines'] == 20 and st['numToneLines'] == 12 and st['firstToneLine'] == 105
    assert st['longestRunWithoutTone'] == 3 and st['tonePresent']

def test_chirp_and_spurs():
    # A tone whose frequency changes within the line, plus the USRP's LO leakage and I/Q image
    rng = np.random.default_rng(5)
    base, _, _ = make_line(0.37, numChan=8, toneScale=0.0, seed=5)
    n = np.arange(512)
    m = (n - 255.5) / 512
    a = 50 * (rng.standard_normal(8) + 1j * rng.standard_normal(8))
    # A small chirp, ~0.08 rad of quadratic phase at the ends of the line (the TSE's is ~0.01 rad): the model's
    # chirp terms are a small-chirp approximation
    chirped = np.outer(a, np.exp(2j * np.pi * (0.37 * n + 0.05 * m * m)))
    spurs = np.outer(a * 0.01, np.exp(2j * np.pi * (0.37 - 1000 * DWELL) * n)) + np.outer(a * 0.02, np.exp(2j * np.pi * (0.37 - 2000 * DWELL) * n))
    line = (base + chirped + spurs).astype(np.complex64)
    def residual(**kw):
        y = line.copy()
        ToneRemover(**kw).process(y, DWELL)
        return np.sqrt(np.mean(np.abs(y - base) ** 2))
    plain, withChirp, full = residual(), residual(chirp=True), residual(chirp=True, spurOffsetsHz=(-1000, -2000))
    # Each part of the model removes its part: the full model leaves much less than the plain one or either
    # part alone (the noise in base is 0.5 per sample; what's left over is the fit's error)
    spursOnly = residual(spurOffsetsHz=(-1000, -2000))
    assert full < plain / 5 and full < withChirp / 3 and full < spursOnly / 3 and full < 0.5

def test_process_acquisition_in_place_with_nominal_frequency():
    import ismrmrd
    from ptone.tests.test_tone_plot import HEADER, F0_HZ
    header = ismrmrd.xsd.CreateFromDocument(HEADER)
    lines, clean = scan_lines([38000.0], [True], seed=6)
    line = lines[0] + tone_model(0.45, np.full(8, 40.0), 512).astype(np.complex64)   # A stronger peak elsewhere
    acq = ismrmrd.Acquisition.from_array(line.astype(np.complex64))
    acq.sample_time_us = DWELL * 1e6
    acq.scan_counter = 42
    acq.read_dir[:] = (1, 0, 0)
    remover = ToneRemover(header, txFreqHz=F0_HZ + 38100.0)               # Nominal 100 Hz from the tone
    assert remover.imagingHalfBand == pytest.approx(0.25)                  # From the header's FOVs
    res = remover.process_acquisition(acq)
    assert res['toneDetected'] and res['freqHz'] == pytest.approx(38000, abs=0.02 / (512 * DWELL))
    assert remover.status()['firstToneLine'] == 42
    expected = line - tone_model(res['freq'], res['amplitudes'], 512)
    np.testing.assert_allclose(acq.data, expected, atol=1e-3)              # Changed in place

def test_remover_speed():
    # A 52-channel x 512-sample line, warm-started: well under a millisecond (about 0.26 ms on the
    # development machine); generous, to allow for slower machines
    import time
    lines, _ = scan_lines(np.full(30, 38000.0), [True] * 30, numChan=52, seed=7)
    remover = ToneRemover()
    remover.process(lines[0], DWELL)
    t = time.perf_counter()
    for line in lines[1:]:
        remover.process(line, DWELL)
    assert (time.perf_counter() - t) / 29 < 2e-3
