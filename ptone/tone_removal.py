# Fitting the pilot tone in a k-space line, and subtracting it.
#
# The pilot tone is a pure sinusoid at one frequency, received by every coil channel with its own amplitude
# and phase.  In one readout line (channels x samples) it is
#     tone[c, n] = a[c] * exp(2j * pi * f * n),   n = 0 .. N-1
# with f the frequency in cycles per sample (the same for every channel; numpy's FFT convention, so it sits
# at bin f * N of np.fft.fft of the line) and a[c] each channel's complex amplitude at the first sample.
#
# fit_tone() estimates f and a from one line, using all channels together:
#   1. Start: the strongest frequency outside the imaging band (the FFT power summed over channels),
#      refined between bins by interpolation; or a given starting frequency.
#   2. Frequency: maximise the tone's power summed over channels, P(f) = sum_c |sum_n y[c,n] exp(-2j pi f n)|^2
#      (the multichannel maximum-likelihood estimate for one sinusoid in white noise), by Newton steps.
#   3. Amplitudes: with f fixed the model is linear, and the least-squares amplitude of each channel is its
#      projection onto the sinusoid: a[c] = (1/N) sum_n y[c,n] exp(-2j pi f n).
# subtract_tone() then removes the fitted tone from the line.
#
# Frequency accuracy matters most: a frequency error df (cycles per sample) leaves a residual of roughly
# |a| * pi * df * N / sqrt(3) per sample, which leaks into every frequency, including the imaging band.

import numpy as np

def tone_model(freq, amplitudes, numSamples):
    """The tone, channels x samples (complex128): amplitudes[c] * exp(2j pi freq n)."""
    n = np.arange(numSamples)
    return np.outer(np.asarray(amplitudes), np.exp(2j * np.pi * freq * n))

def tone_amplitudes(data, freq):
    """Least-squares complex amplitude of a tone at freq (cycles per sample) on each channel, at sample 0."""
    data = np.asarray(data)
    n = np.arange(data.shape[1])
    return data @ np.exp(-2j * np.pi * freq * n) / data.shape[1]

def start_frequency(data, imagingHalfBand=0.25):
    """
    A starting frequency (cycles per sample) for the tone: the FFT bin with the most power summed over
    channels, among bins outside the imaging band (|f| >= imagingHalfBand; 0.25 for 2x readout
    oversampling), then interpolated between bins from the complex FFT values of all channels (Jacobsen's
    estimator, combined over channels weighted by power).
    """
    data = np.asarray(data)
    numSamples = data.shape[1]
    spec = np.fft.fft(data, axis=1)
    freqs = np.fft.fftfreq(numSamples)
    power = np.sum(np.abs(spec) ** 2, axis=0)
    outside = np.abs(freqs) >= imagingHalfBand
    if not np.any(outside):
        outside[:] = True
    k = np.flatnonzero(outside)[np.argmax(power[outside])]
    xm, x0, xp = spec[:, (k - 1) % numSamples], spec[:, k], spec[:, (k + 1) % numSamples]
    num = np.sum(np.conj(x0) * (xm - xp))
    den = np.sum(np.conj(x0) * (2 * x0 - xm - xp))
    delta = float(np.real(num / den)) if den != 0 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(((k + delta) / numSamples + 0.5) % 1.0 - 0.5)

def refine_frequency(data, freq, maxIterations=30, tolerance=1e-10):
    """
    Refine the tone's frequency (cycles per sample) by Newton steps on P(f), the tone's power summed over
    channels.  Steps are limited to a quarter of a bin, so a start within the main lobe of the tone's peak
    converges to it.  Returns (freq, number of iterations).
    """
    data = np.asarray(data)
    numSamples = data.shape[1]
    n = np.arange(numSamples, dtype=float)
    m = n - (numSamples - 1) / 2            # Centred sample index: better conditioned derivatives
    maxStep = 0.25 / numSamples
    for iteration in range(1, maxIterations + 1):
        e = np.exp(-2j * np.pi * freq * n)
        sums = data @ np.stack([e, m * e, m * m * e], axis=1)     # channels x 3: sum y e, sum m y e, sum m^2 y e
        s0, s1, s2 = sums[:, 0], sums[:, 1], sums[:, 2]
        # P = sum |S0|^2 with S0(f) = sum_n y exp(-2j pi f n); the m-weighted sums give its derivatives
        # (the centring only changes S0 by a phase factor, which P doesn't see)
        d1 = 4 * np.pi * np.sum(np.imag(np.conj(s0) * s1))
        d2 = 2 * (2 * np.pi) ** 2 * np.sum(np.abs(s1) ** 2 - np.real(np.conj(s0) * s2))
        step = -d1 / d2 if d2 < 0 else np.sign(d1) * maxStep
        step = float(np.clip(step, -maxStep, maxStep))
        freq = freq + step
        if abs(step) < tolerance:
            break
    return float((freq + 0.5) % 1.0 - 0.5), iteration

def peak_to_noise_db(data, freq, amplitudes, imagingHalfBand=0.25):
    """
    How far the fitted tone stands above the rest of the spectrum outside the imaging band, in dB: the
    tone's power summed over channels, sum_c |N a_c|^2, over the median FFT power (summed over channels) of
    the bins outside the imaging band.  Tens of dB with the tone; a few dB when there's no tone (the fit
    then just finds the largest noise peak).
    """
    data = np.asarray(data)
    numSamples = data.shape[1]
    power = np.sum(np.abs(np.fft.fft(data, axis=1)) ** 2, axis=0)
    outside = np.abs(np.fft.fftfreq(numSamples)) >= imagingHalfBand
    floor = np.median(power[outside]) if np.any(outside) else np.median(power)
    tonePower = np.sum(np.abs(numSamples * np.asarray(amplitudes)) ** 2)
    return float(10 * np.log10(tonePower / floor)) if floor > 0 else float('inf')

def fit_tone(data, freqStart=None, imagingHalfBand=0.25):
    """
    Fit the pilot tone in one line (channels x samples).  freqStart: a starting frequency in cycles per
    sample (default: start_frequency()).  imagingHalfBand: the imaging band is |f| < this (cycles per
    sample; 0.25 for 2x readout oversampling), where the starting search doesn't look.  Returns a dict:
      freq           frequency, cycles per sample (in [-0.5, 0.5))
      amplitudes     complex amplitude of each channel at the first sample
      peakToNoiseDb  how far the tone stands above the spectrum outside the imaging band (peak_to_noise_db())
      iterations     Newton iterations used
      freqStart      the starting frequency
    """
    data = np.asarray(data)
    if freqStart is None:
        freqStart = start_frequency(data, imagingHalfBand)
    freq, iterations = refine_frequency(data, freqStart)
    amplitudes = tone_amplitudes(data, freq)
    return {'freq': freq, 'amplitudes': amplitudes,
            'peakToNoiseDb': peak_to_noise_db(data, freq, amplitudes, imagingHalfBand),
            'iterations': iterations, 'freqStart': float(freqStart)}

def subtract_tone(data, fit):
    """The line with the fitted tone subtracted (a new array, same dtype as data)."""
    data = np.asarray(data)
    return (data - tone_model(fit['freq'], fit['amplitudes'], data.shape[1])).astype(data.dtype)

def band_power_reduction_db(before, after, freq, halfWidthBins=3):
    """
    How much the power within halfWidthBins FFT bins of freq fell from `before` to `after` (one channel
    or channels x samples), in dB: a measure of how well the tone was removed.
    """
    before, after = np.atleast_2d(before), np.atleast_2d(after)
    numSamples = before.shape[1]
    binDistance = (np.arange(numSamples) - freq * numSamples + numSamples / 2) % numSamples - numSamples / 2
    band = np.abs(binDistance) <= halfWidthBins
    p0 = np.sum(np.abs(np.fft.fft(before, axis=1)[:, band]) ** 2)
    p1 = np.sum(np.abs(np.fft.fft(after, axis=1)[:, band]) ** 2)
    return float(10 * np.log10(p0 / p1)) if p1 > 0 else float('inf')
