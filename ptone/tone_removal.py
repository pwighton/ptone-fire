# Removing the pilot tone from k-space lines: subtracting the tone estimated by ptone/tone_estimation.py.
#
# subtract_tone() removes a fitted tone (fit_tone()) from a line.  ToneRemover does it live, line by line: it
# estimates each line's tone with ToneEstimator and, if the line has it, subtracts it in place, optionally with
# extra model terms (a chirp, and extra tones such as the USRP's LO leakage and I/Q image).
#
# Frequency accuracy matters most: a frequency error df (cycles per sample) leaves a residual of roughly
# |a| * pi * df * N / sqrt(3) per sample, which leaks into every frequency, including the imaging band.

import os
import sys

import numpy as np

try:
    from ptone.tone_estimation import ToneEstimator, tone_model
except ImportError:                 # Run from ptone/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ptone.tone_estimation import ToneEstimator, tone_model

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

# ----- Real-time removal, line by line -----------------------------------------------------------

class ToneRemover(ToneEstimator):
    """
    Removes the pilot tone from k-space lines one at a time, as they arrive: ToneEstimator's estimate for each
    line (see its docstring), then, if the line has the tone, the tone subtracted in place.  Lines without the
    tone are left unchanged.

    The subtracted model is the tone (each channel's least-squares amplitude at the estimated frequency), plus,
    optionally:
      chirp:          a linear frequency change within the line (the TSE's is ~2 Hz over a line)
      spurOffsetsHz:  extra tones at these offsets from it (Hz), e.g. (-1000, -2000) for the USRP's LO leakage
                      and I/Q image when it transmits 1 kHz above its LO
    fitted together by least squares.  Other arguments: as ToneEstimator.
    """

    def __init__(self, mrdHeader=None, txFreqHz=None, chirp=False, spurOffsetsHz=(), **estimatorArgs):
        super().__init__(mrdHeader, txFreqHz, **estimatorArgs)
        self.chirp = bool(chirp)
        self.spurOffsetsHz = tuple(float(x) for x in spurOffsetsHz)

    def _columns(self, freq, numSamples, dwellS):
        # Model columns (K x N): the tone, then optionally the chirp terms and extra tones
        n = np.arange(numSamples)
        e = np.exp(2j * np.pi * freq * n)
        cols = [e]
        if self.chirp:
            m = (n - (numSamples - 1) / 2) / numSamples
            cols += [e * m, e * m * m]
        for offsetHz in self.spurOffsetsHz:
            cols.append(np.exp(2j * np.pi * (freq + offsetHz * dwellS) * n))
        return np.array(cols)

    def process(self, data, dwellS, nominalHz=None, lineId=None, subtract=True):
        """
        Estimate the tone in one line (ToneEstimator.estimate(); same arguments and result) and, if the line has
        it and subtract is true, subtract it from data (channels x samples) in place.  With extra model terms
        the result's amplitudes are the tone's, fitted together with them.
        """
        result = self.estimate(data, dwellS, nominalHz, lineId)
        if not (result['toneDetected'] and subtract):
            return result
        cols = self._columns(result['freq'], data.shape[1], dwellS)
        if len(cols) == 1:
            coefs = result['amplitudes'][:, None]
        else:
            y = np.asarray(data, dtype=np.complex128)
            gram = cols.conj() @ cols.T                                       # K x K
            coefs = np.linalg.solve(gram.T, (y @ cols.conj().T).T).T            # channels x K, least squares
            result['amplitudes'] = coefs[:, 0]
        # The phase ramps are computed in double precision; the subtraction is done in the data's precision
        data -= coefs.astype(data.dtype) @ cols.astype(data.dtype)
        return result

    def process_acquisition(self, acq, subtract=True):
        """process() for an ismrmrd.Acquisition, modifying acq.data in place; its scan counter is the line id."""
        return self.process(acq.data, acq.sample_time_us * 1e-6, self.nominal_freq_hz(acq), lineId=int(acq.scan_counter),
                            subtract=subtract)
