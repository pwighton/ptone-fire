# Tests for ptone/tone_removal.py: fitting the pilot tone in a k-space line and subtracting it

import numpy as np
import pytest

from ptone.tone_removal import (band_power_reduction_db, fit_tone, refine_frequency, start_frequency, subtract_tone,
                                tone_amplitudes, tone_model)

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

def test_subtract_tone_leaves_the_rest():
    line, withoutTone, amplitudes = make_line()
    filtered = subtract_tone(line, fit_tone(line))
    assert filtered.dtype == np.complex64 and filtered.shape == line.shape
    # What's left is the line without the tone, apart from the fit's small error (noise-limited)
    residual = np.sqrt(np.mean(np.abs(filtered - withoutTone) ** 2))
    assert residual < 0.02 * np.sqrt(np.mean(np.abs(amplitudes) ** 2))
    assert band_power_reduction_db(line, filtered, fit_tone(line)['freq']) > 30

def test_peak_to_noise_with_and_without_tone():
    line, withoutTone, _ = make_line()
    assert fit_tone(line)['peakToNoiseDb'] > 30
    assert fit_tone(withoutTone.astype(np.complex64))['peakToNoiseDb'] < 15

def test_band_power_reduction():
    line, _, _ = make_line()
    assert band_power_reduction_db(line, line, 0.3787123) == pytest.approx(0.0)
    assert band_power_reduction_db(line, line / 10, 0.3787123) == pytest.approx(20.0)
