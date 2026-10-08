# Tests for ptone/tone_estimation.py: estimating the pilot tone in a k-space line, and line by line

import numpy as np
import pytest

from ptone.tone_estimation import (ToneEstimator, fit_tone, refine_frequency, start_frequency, tone_amplitudes, tone_model,
                                   tone_quality)

def make_line(freq=0.3787123, numChan=16, numSamples=512, toneScale=5.0, imageScale=30.0, noise=0.5, seed=0):
    # One readout line (channels x samples, complex64): an image-like signal confined to the imaging band
    # (the central half of the spectrum, as with 2x oversampling), white noise, and the tone.  Returns
    # (line, the line without the tone, the tone's amplitudes)
    rng = np.random.default_rng(seed)
    band = rng.standard_normal((numChan, numSamples // 2)) + 1j * rng.standard_normal((numChan, numSamples // 2))
    spectrum = np.pad(band, ((0, 0), (numSamples // 4, numSamples // 4)))
    image = imageScale * np.fft.ifft(np.fft.ifftshift(spectrum, axes=1), axis=1)
    noiseLine = noise * (rng.standard_normal((numChan, numSamples)) + 1j * rng.standard_normal((numChan, numSamples)))
    amplitudes = toneScale * (rng.standard_normal(numChan) + 1j * rng.standard_normal(numChan))
    withoutTone = image + noiseLine
    line = (withoutTone + tone_model(freq, amplitudes, numSamples)).astype(np.complex64)
    return line, withoutTone, amplitudes

def test_tone_model_and_amplitudes():
    a = np.array([1 + 2j, -0.5j])
    tone = tone_model(0.1, a, 64)
    assert tone.shape == (2, 64)
    np.testing.assert_allclose(tone[:, 0], a)                      # Amplitudes are at the first sample
    np.testing.assert_allclose(tone[:, 10], a * np.exp(2j * np.pi * 0.1 * 10))
    np.testing.assert_allclose(tone_amplitudes(tone, 0.1), a)      # Exact for a pure tone

@pytest.mark.parametrize('freq', [0.3787123, -0.4, 0.375, -0.30001, 0.49])
def test_fit_finds_frequency_and_amplitudes(freq):
    line, _, amplitudes = make_line(freq)
    fit = fit_tone(line)
    assert fit['freq'] == pytest.approx(freq, abs=1e-3 / 512)      # Within 0.001 of a bin (noise-limited)
    assert np.max(np.abs(fit['amplitudes'] - amplitudes)) < 0.1 * np.median(np.abs(amplitudes))
    assert fit['iterations'] < 10

def test_start_frequency_is_within_the_peak():
    # The interpolated start is close enough for the Newton steps (within a fraction of a bin)
    line, _, _ = make_line(0.3787123)
    assert abs(start_frequency(line) - 0.3787123) * 512 < 0.1

def test_start_frequency_ignores_imaging_band():
    # A strong component inside the imaging band isn't taken for the tone
    line, _, _ = make_line(0.4, toneScale=1.0)
    line = line + (200 * np.exp(2j * np.pi * 0.05 * np.arange(512)))[None, :].astype(np.complex64)
    assert start_frequency(line) == pytest.approx(0.4, abs=1 / 512)
    assert start_frequency(line, imagingHalfBand=0.0) == pytest.approx(0.05, abs=1 / 512)

def test_refine_from_a_given_start():
    line, _, _ = make_line(0.3787123)
    freq, _ = refine_frequency(line, 0.3787123 + 0.3 / 512)        # A third of a bin away
    assert freq == pytest.approx(0.3787123, abs=1e-3 / 512)
    assert fit_tone(line, freqStart=0.3787123 - 0.3 / 512)['freqStart'] == pytest.approx(0.3787123 - 0.3 / 512)

def test_peak_to_noise_with_and_without_tone():
    line, withoutTone, _ = make_line()
    assert fit_tone(line)['peakToNoiseDb'] > 30
    assert fit_tone(withoutTone.astype(np.complex64))['peakToNoiseDb'] < 15

# ----- ToneEstimator: line by line -------------------------------------------------------------

DWELL = 9.8e-6

def scan_lines(freqsHz, toneOn, numChan=8, seed=0, dwell=DWELL, toneScale=5.0):
    # Lines with the tone at the given frequencies (Hz from the band centre), or without it where toneOn is
    # False.  The channel pattern stays the same from line to line, as the real tone's nearly does.  Returns
    # (lines, lines without the tone)
    rng = np.random.default_rng(seed)
    pattern = toneScale * (rng.standard_normal(numChan) + 1j * rng.standard_normal(numChan))
    lines, clean = [], []
    for i, (fHz, on) in enumerate(zip(freqsHz, toneOn)):
        line, withoutTone, _ = make_line(fHz * dwell, numChan=numChan, toneScale=0.0, seed=seed * 1000 + i)
        if on:
            line = (line + tone_model(fHz * dwell, pattern * np.exp(1j * rng.uniform(0, 0.1)), line.shape[1])).astype(np.complex64)
        lines.append(line)
        clean.append(withoutTone)
    return lines, clean

def test_tone_quality():
    assert tone_quality(25.0) == pytest.approx(0.5)
    assert tone_quality(50.0) > 0.99 and tone_quality(5.0) < 0.01
    assert tone_quality(30.0, thresholdDb=30.0) == pytest.approx(0.5)

def test_estimator_follows_line_to_line_changes():
    # The frequency jumps between lines by up to a fifth of a bin (like the TSE's per-slice levels) and drifts
    rng = np.random.default_rng(1)
    binHz = 1 / (512 * DWELL)
    freqsHz = 38000 + np.cumsum(rng.uniform(-0.2, 0.2, 40)) * binHz
    lines, clean = scan_lines(freqsHz, [True] * 40)
    estimator = ToneEstimator()
    for i, line in enumerate(lines):
        original = line.copy()
        res = estimator.estimate(line, DWELL, lineId=i)
        np.testing.assert_array_equal(line, original)                    # The estimator never changes the data
        assert res['toneDetected'] and res['searched'] == (i == 0)       # Only the first line needs a search
        assert res['freqHz'] == pytest.approx(freqsHz[i], abs=0.02 * binHz)
        assert res['quality'] > 0.99
    assert estimator.status()['numToneLines'] == 40 and estimator.status()['tonePresent']

def test_lines_without_the_tone_and_status():
    # No tone for the first 5 lines (transmitter not started) and lines 12-14 (a gap)
    toneOn = [i >= 5 and not 12 <= i <= 14 for i in range(20)]
    lines, _ = scan_lines(np.full(20, 38000.0), toneOn, seed=2)
    estimator = ToneEstimator()
    lastFreq = []
    for i, line in enumerate(lines):
        res = estimator.estimate(line, DWELL, lineId=100 + i)
        assert res['toneDetected'] == toneOn[i]
        if not toneOn[i]:
            assert res['quality'] < 0.5
        lastFreq.append(estimator.lastFreqHz)
    assert lastFreq[11] == lastFreq[12] == lastFreq[14]                  # Lines without the tone aren't warm starts
    st = estimator.status()
    assert st['numLines'] == 20 and st['numToneLines'] == 12 and st['firstToneLine'] == 105
    assert st['longestRunWithoutTone'] == 3 and st['tonePresent']

def test_strong_signal_outside_the_imaging_band_is_not_a_tone():
    # Like a central k-space line whose own signal reaches outside the imaging band: broadband, not a peak
    rng = np.random.default_rng(3)
    line = (300 * (rng.standard_normal((8, 512)) + 1j * rng.standard_normal((8, 512)))).astype(np.complex64)
    line[:, 250:262] += 5000                                              # A strong echo
    res = ToneEstimator().estimate(line.copy(), DWELL)
    assert not res['toneDetected'] and res['localDb'] < 15

def test_search_near_the_nominal_frequency():
    # A stronger peak elsewhere outside the imaging band: without the nominal frequency the search takes it;
    # with it, the search looks near the nominal frequency and finds the tone
    lines, _ = scan_lines([38000.0], [True], seed=4)
    other = 0.45 / DWELL
    line = (lines[0] + tone_model(0.45, np.full(8, 40.0), 512)).astype(np.complex64)
    assert ToneEstimator().estimate(line.copy(), DWELL)['freqHz'] == pytest.approx(other, abs=1)
    assert ToneEstimator().estimate(line.copy(), DWELL, nominalHz=38150.0)['freqHz'] == pytest.approx(38000, abs=1)

def test_different_dwell_times():
    # The warm start is kept in Hz, so a line with another dwell time starts in the right place
    estimator = ToneEstimator()
    for i, dwell in enumerate([9.8e-6, 16.3e-6, 9.8e-6, 16.3e-6]):
        # 28 kHz is outside the imaging band at both dwell times (0.27 and 0.46 cycles per sample)
        lines, clean = scan_lines([28000.0], [True], seed=10 + i, dwell=dwell)
        res = estimator.estimate(lines[0], dwell)
        # Within 0.02 of a bin (bins are 199 and 120 Hz here; 8 channels limit the precision)
        assert res['toneDetected'] and res['freqHz'] == pytest.approx(28000, abs=0.02 / (512 * dwell))
        assert res['searched'] == (i == 0)

def test_estimate_acquisition_with_nominal_frequency():
    import ismrmrd
    from ptone.tests.test_tone_plot import HEADER, F0_HZ
    header = ismrmrd.xsd.CreateFromDocument(HEADER)
    lines, _ = scan_lines([38000.0], [True], seed=6)
    line = (lines[0] + tone_model(0.45, np.full(8, 40.0), 512)).astype(np.complex64)   # A stronger peak elsewhere
    acq = ismrmrd.Acquisition.from_array(line)
    acq.sample_time_us = DWELL * 1e6
    acq.scan_counter = 42
    acq.read_dir[:] = (1, 0, 0)
    estimator = ToneEstimator(header, txFreqHz=F0_HZ + 38100.0)             # Nominal 100 Hz from the tone
    assert estimator.imagingHalfBand == pytest.approx(0.25)                 # From the header's FOVs
    assert estimator.nominal_freq_hz(acq) == pytest.approx(38100.0)
    res = estimator.estimate_acquisition(acq)
    assert res['toneDetected'] and res['freqHz'] == pytest.approx(38000, abs=0.02 / (512 * DWELL))
    assert estimator.status()['firstToneLine'] == 42
    np.testing.assert_array_equal(acq.data, line)                           # Unchanged

def test_estimator_speed():
    # A 52-channel x 512-sample line, warm-started: well under a millisecond; generous, to allow for slower machines
    import time
    lines, _ = scan_lines(np.full(30, 38000.0), [True] * 30, numChan=52, seed=7)
    estimator = ToneEstimator()
    estimator.estimate(lines[0], DWELL)
    t = time.perf_counter()
    for line in lines[1:]:
        estimator.estimate(line, DWELL)
    assert (time.perf_counter() - t) / 29 < 2e-3
