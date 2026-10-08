# Estimating the pilot tone in k-space lines: its frequency, each channel's amplitude and phase, and whether a
# line has the tone at all.  ptone/tone_removal.py subtracts what's estimated here.
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
#
# ToneEstimator does this live, line by line: it decides whether each line has the tone, and if so fits it,
# warm-started from the last line that had it; see its docstring.

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

def start_frequency(data, imagingHalfBand=0.25, searchCentre=None, searchHalfWidthBins=5):
    """
    A starting frequency (cycles per sample) for the tone: the FFT bin with the most power summed over
    channels, among bins outside the imaging band (|f| >= imagingHalfBand; 0.25 for 2x readout
    oversampling), or, if searchCentre (cycles per sample, e.g. the nominal frequency) is given, among bins
    within searchHalfWidthBins of it; then interpolated between bins from the complex FFT values of all
    channels (Jacobsen's estimator, combined over channels weighted by power).
    """
    data = np.asarray(data)
    numSamples = data.shape[1]
    spec = np.fft.fft(data, axis=1)
    freqs = np.fft.fftfreq(numSamples)
    power = np.sum(np.abs(spec) ** 2, axis=0)
    if searchCentre is not None:
        binDistance = np.abs(((freqs - searchCentre) + 0.5) % 1.0 - 0.5) * numSamples
        candidates = binDistance <= searchHalfWidthBins
    else:
        candidates = np.abs(freqs) >= imagingHalfBand
    if not np.any(candidates):
        candidates[:] = True
    k = np.flatnonzero(candidates)[np.argmax(power[candidates])]
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

def tone_quality(localDb, thresholdDb=25.0):
    """
    A 0-1 measure of how clearly a line has the tone, from local_db(): 1 / (1 + 10^((thresholdDb - localDb) / 10)).
    0.5 at thresholdDb (where ToneEstimator decides there's a tone), ~1 for a clear tone, ~0 without one.
    """
    return float(1.0 / (1.0 + 10 ** ((thresholdDb - localDb) / 10.0)))

class ToneEstimator:
    """
    Estimates the pilot tone in k-space lines one at a time, as they arrive (ToneRemover, in
    ptone/tone_removal.py, also subtracts it).

    For each line (estimate(), or estimate_acquisition() for an ismrmrd.Acquisition):
      1. Starting frequency: the frequency of the last line that had the tone (the frequency changes a
         little from line to line, so each line is fitted, but it stays within a fraction of a bin).  If
         there is no such line yet, a search: the strongest FFT bin within searchHalfWidthBins of the
         nominal frequency (from the transmit frequency, if known), else outside the imaging band.
      2. Refine the frequency on this line (Newton steps, all channels together; refine_frequency()).
      3. Is the tone there?  local_db: the tone's power summed over channels, over the mean power of the
         projections sideBins FFT bins either side of it.  A pure tone contributes nothing at whole-bin
         offsets, so this compares it with the noise and image signal around it; a tone gives ~40-60 dB, a
         line without one ~0-15 dB (recorded data, 2026-10-08).  If below thresholdDb after a warm start, the
         search of step 1 is tried once; if still below, the line has no tone, and isn't used as a warm start.
      4. If the tone is there: each channel's amplitude by least squares, and the frequency is remembered (in
         Hz, so lines with different dwell times can follow each other).

    status() reports, for the scan so far: lines processed, lines with the tone, the first line with it, and
    the longest run of lines without it since it appeared.

    imagingHalfBand: the imaging band is |f| < this (cycles per sample).  Default from the MRD header (recon
    FOV / encoded FOV / 2) if given, else 0.25 (2x readout oversampling).
    txFreqHz: the transmit frequency, for the nominal frequency (needs the header; see ptone/tx_frequency.py).
    """

    def __init__(self, mrdHeader=None, txFreqHz=None, imagingHalfBand=None, thresholdDb=25.0, searchHalfWidthBins=5,
                 sideBins=(3, 4, 5, 6), maxIterations=10):
        self.mrdHeader = mrdHeader
        self.txFreqHz = txFreqHz
        if imagingHalfBand is None:
            imagingHalfBand = 0.25
            try:
                enc = mrdHeader.encoding[0]
                imagingHalfBand = 0.5 * enc.reconSpace.fieldOfView_mm.x / enc.encodedSpace.fieldOfView_mm.x
            except (AttributeError, IndexError, TypeError, ZeroDivisionError):
                pass
        self.imagingHalfBand = float(imagingHalfBand)
        self.thresholdDb = float(thresholdDb)
        self.searchHalfWidthBins = searchHalfWidthBins
        self.sideBins = tuple(sideBins)
        self.maxIterations = int(maxIterations)
        self._sideTables = {}
        self.lastFreqHz = None             # Frequency of the last line with the tone (Hz from the band centre)
        self.numLines = 0
        self.numToneLines = 0
        self.firstToneLine = None          # Line id (see estimate()) of the first line with the tone
        self.runWithout = 0                # Lines without the tone since the last line with it
        self.longestRunWithout = 0         # Longest such run, after the tone first appeared

    def nominal_freq_hz(self, acq):
        """The tone's nominal frequency in acq's readout (Hz from the band centre), or None if unknown."""
        if self.txFreqHz is None or self.mrdHeader is None:
            return None
        try:
            from ptone.tx_frequency import ptone_readout_offset
        except ImportError:
            from tx_frequency import ptone_readout_offset
        try:
            return ptone_readout_offset(self.txFreqHz, self.mrdHeader, acq)
        except (AttributeError, IndexError, TypeError):
            return None

    def _side_table(self, numSamples):
        # exp(-2j pi k n / N) for k = 0 and the side bins, N x (1 + 2 * len(sideBins)); cached per N
        table = self._sideTables.get(numSamples)
        if table is None:
            n = np.arange(numSamples)
            k = np.array((0,) + tuple(-b for b in self.sideBins) + self.sideBins, dtype=float)
            table = np.exp(-2j * np.pi * np.outer(n, k) / numSamples)
            self._sideTables[numSamples] = table
        return table

    def local_db(self, data, freq, returnAmplitudes=False):
        """
        The tone's power at freq over the mean power sideBins bins either side (see the class docstring).
        With returnAmplitudes, also each channel's least-squares amplitude at freq (the same projection).
        """
        numSamples = data.shape[1]
        e = np.exp(-2j * np.pi * freq * np.arange(numSamples))
        proj = data @ (e[:, None] * self._side_table(numSamples))      # channels x offsets
        power = np.sum(np.abs(proj) ** 2, axis=0)
        side = np.mean(power[1:])
        localDb = float(10 * np.log10(power[0] / side)) if side > 0 else float('inf')
        return (localDb, proj[:, 0] / numSamples) if returnAmplitudes else localDb

    def _search(self, data, dwellS, nominalHz):
        centre = None if nominalHz is None else ((nominalHz * dwellS + 0.5) % 1.0 - 0.5)
        return start_frequency(data, self.imagingHalfBand, searchCentre=centre, searchHalfWidthBins=self.searchHalfWidthBins)

    def estimate(self, data, dwellS, nominalHz=None, lineId=None):
        """
        Estimate the tone in one line: data (channels x samples; not modified), dwellS (s per sample), nominalHz
        (the tone's nominal frequency, Hz from the band centre, or None), lineId (e.g. the scan counter, for
        status()).  Returns a dict:
          toneDetected  whether the line has the tone
          freqHz, freq  the tone's frequency: Hz from the band centre, and cycles per sample
          amplitudes    each channel's complex amplitude at the first sample
          localDb       the detection measure (see the class docstring); quality: tone_quality() of it
          iterations    Newton iterations; searched: whether the starting frequency came from a search
        """
        self.numLines += 1
        y = np.asarray(data, dtype=np.complex128)      # One conversion; the fit works in double precision
        searched = self.lastFreqHz is None
        start = self._search(y, dwellS, nominalHz) if searched else ((self.lastFreqHz * dwellS + 0.5) % 1.0 - 0.5)
        freq, iterations = refine_frequency(y, start, maxIterations=self.maxIterations)
        localDb, amplitudes = self.local_db(y, freq, returnAmplitudes=True)
        if localDb < self.thresholdDb and not searched:
            # The warm start found no tone: try a search once
            freq2, it2 = refine_frequency(y, self._search(y, dwellS, nominalHz), maxIterations=self.maxIterations)
            local2, amp2 = self.local_db(y, freq2, returnAmplitudes=True)
            iterations += it2
            searched = True
            if local2 > localDb:
                freq, localDb, amplitudes = freq2, local2, amp2
        detected = localDb >= self.thresholdDb
        result = {'toneDetected': bool(detected), 'freq': float(freq), 'freqHz': float(freq / dwellS),
                  'amplitudes': amplitudes, 'localDb': localDb, 'quality': tone_quality(localDb, self.thresholdDb),
                  'iterations': iterations, 'searched': searched}
        if not detected:
            if self.firstToneLine is not None:
                self.runWithout += 1
                self.longestRunWithout = max(self.longestRunWithout, self.runWithout)
            return result
        self.numToneLines += 1
        self.runWithout = 0
        if self.firstToneLine is None:
            self.firstToneLine = lineId if lineId is not None else self.numLines - 1
        self.lastFreqHz = freq / dwellS
        return result

    def estimate_acquisition(self, acq):
        """estimate() for an ismrmrd.Acquisition; its scan counter is the line id."""
        return self.estimate(acq.data, acq.sample_time_us * 1e-6, self.nominal_freq_hz(acq), lineId=int(acq.scan_counter))

    def status(self):
        """The tone over the scan so far (see the class docstring)."""
        return {'numLines': self.numLines, 'numToneLines': self.numToneLines, 'firstToneLine': self.firstToneLine,
                'linesWithoutToneSinceLast': self.runWithout if self.firstToneLine is not None else None,
                'longestRunWithoutTone': self.longestRunWithout, 'tonePresent': self.firstToneLine is not None and self.runWithout == 0,
                'lastFreqHz': self.lastFreqHz}
