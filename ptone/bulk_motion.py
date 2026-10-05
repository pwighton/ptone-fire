# Real-time bulk head motion score from the pilot tone.
#
# A score of how much the head has moved from one position to another, computed line by line from the
# estimates pilottone.py already calculates (analyze_line()), using only lines already seen, so it can run
# live.  It ignores slow drift and periodic changes (from the sequence, breathing and the heartbeat).
#
# Methods share one interface, so others can be added later:
#   method = create_bulk_motion('medianFilter', trMs, {'minWindowS': 5, 'minQuality': 0})
#   score = method.update(timeMs, relativePhase, relativeAmplitude, quality)    # once per analysed line
# update() returns the current score, or None while the method isn't ready or for lines it skips.  Every
# method receives every line, with relative phase and amplitude for all channels and the line's quality,
# and decides itself which to use.
#
# See ~/lcn/projects/ptone-fire/20261005-head-motion-metric-development.md for the reasoning.

import collections
import math

import numpy as np

class MedianFilter:
    """
    Bulk motion score from each channel's relative phase, smoothed with a median over a time window.

    Relative phase behaves like a position: it depends on where the head is.  So the head moving from
    one position to another shifts many channels' relative phase from one level to another.  Each time
    the score is computed, each channel's median over the latest window is compared with its median over
    the window before it:
    - The window is the smallest whole number of TRs at least minWindowS long.  The sequence makes the
      signal wobble slightly in a pattern that repeats once per TR; whole cycles make that wobble the
      same in every window, so it doesn't look like a change.  At least minWindowS (e.g. 5 s) also spans
      a breath and several heartbeats, to damp those.
    - A median switches cleanly to a new level when the head moves, and ignores odd outlying lines.
    - Each channel's change is divided by its noise (estimated from the data, see below) and by the
      expected spread of a difference between two window medians for the number of lines in the windows.
      Channels are combined as a root-mean-square.  If the remaining variation were just random noise, a
      still head would score about 1 whatever the sequence; a movement scores higher, for about one
      window.
    - Noise: each channel's typical line-to-line change over the last NOISE_HORIZON_MS, as a median, which
      a movement's few large changes barely affect.  So nothing assumes the head is still at the start.

    Parameters:
      trMs:       the sequence TR in ms (MRD header sequenceParameters.TR[0]), or None to use minWindowS
      minWindowS: shortest window in seconds (default 5)
      minQuality: lines whose quality is below this are skipped (default 0: none skipped)

    Relative amplitude is ignored.  Relative phase isn't unwrapped: it relies on pilottone.py setting the
    phase midpoint from a line with the tone (ptoneQualityThreshold), which keeps it continuous in practice.
    """

    name = 'medianFilter'

    # Internal settings
    UPDATE_MS = 100.0               # How often the score is recomputed
    NOISE_UPDATE_MS = 1000.0        # How often the noise estimates are recomputed
    NOISE_HORIZON_MS = 60000.0      # Line-to-line changes from this far back are used for the noise
    MIN_NOISE_DIFFS = 50            # Line-to-line changes needed before there's a noise estimate
    MIN_LINES_PER_WINDOW = 5        # Fewer lines than this in either window: no score
    MAD_TO_SD = 1.4826              # Standard deviation / median absolute deviation, for normal noise
    MEDIAN_SE_FACTOR = math.sqrt(math.pi / 2)   # Standard error of a median / (sd / sqrt(n)), large n

    def __init__(self, trMs, minWindowS=5.0, minQuality=0.0):
        minWindowS = float(minWindowS)
        if minWindowS <= 0:
            raise ValueError("minWindowS must be positive (got %s)" % minWindowS)
        self.trMs = float(trMs) if (trMs is not None and float(trMs) > 0) else None
        self.minWindowS = minWindowS
        self.minQuality = float(minQuality)
        self.windowMs = self.window_ms(self.trMs, minWindowS)

        self.numChan = None
        self.prevPhase = None                               # Last accepted line's relative phase
        self.prevTimeMs = None
        self.firstTimeMs = None
        self.lines = collections.deque()                    # (timeMs, relative phase) within 2 windows
        self.diffs = collections.deque()                    # (timeMs, |line-to-line change|) within the noise horizon
        self.noise = None                                   # Per-channel noise (sd of one line's phase)
        self.noiseTimeMs = None
        self.score = None
        self.scoreTimeMs = None

    @staticmethod
    def window_ms(trMs, minWindowS):
        """The smallest whole number of TRs that's at least minWindowS long, in ms (minWindowS without a TR)."""
        minWindowMs = minWindowS * 1000.0
        if trMs is None:
            return minWindowMs
        numTr = math.ceil(minWindowMs / trMs - 1e-9)
        return numTr * trMs

    def update(self, timeMs, relativePhase, relativeAmplitude, quality):
        """Add one analysed line.  Returns the current score, or None if not ready or the line was skipped."""
        if quality < self.minQuality:
            return None
        phase = np.asarray(relativePhase, dtype=float)
        if self.numChan is None:
            self.numChan = phase.shape[0]
        elif phase.shape[0] != self.numChan:
            return None
        timeMs = float(timeMs)

        # Relative phase is used as pilottone.py saves it: within +/- pi of the midpoint set from the first
        # line with the tone (ptoneQualityThreshold), which keeps it continuous in practice.  It isn't
        # unwrapped here (a 2*pi wrap would look like a large change)
        if self.prevPhase is not None:
            self.diffs.append((timeMs, np.abs(phase - self.prevPhase)))
        else:
            self.firstTimeMs = timeMs
        self.prevPhase = phase
        self.prevTimeMs = timeMs
        self.lines.append((timeMs, phase))

        # Only keep what the windows and noise estimate need
        while self.lines and self.lines[0][0] < timeMs - 2 * self.windowMs:
            self.lines.popleft()
        while self.diffs and self.diffs[0][0] < timeMs - self.NOISE_HORIZON_MS:
            self.diffs.popleft()

        if self.noiseTimeMs is None or timeMs - self.noiseTimeMs >= self.NOISE_UPDATE_MS:
            self.update_noise(timeMs)
        if self.scoreTimeMs is None or timeMs - self.scoreTimeMs >= self.UPDATE_MS:
            self.score = self.compute_score(timeMs)
            self.scoreTimeMs = timeMs
        return self.score

    def update_noise(self, timeMs):
        if len(self.diffs) >= self.MIN_NOISE_DIFFS:
            absDiffs = np.stack([d for _, d in self.diffs])
            # Line-to-line changes of independent noise with sd s have sd s*sqrt(2)
            self.noise = np.median(absDiffs, axis=0) * self.MAD_TO_SD / math.sqrt(2)
        self.noiseTimeMs = timeMs

    def compute_score(self, timeMs):
        # None until there are two complete windows and a noise estimate
        if self.noise is None or timeMs - self.firstTimeMs < 2 * self.windowMs:
            return None
        times = np.array([t for t, _ in self.lines])
        phases = np.stack([p for _, p in self.lines])
        latest = times > timeMs - self.windowMs
        previous = ~latest & (times > timeMs - 2 * self.windowMs)
        nLatest, nPrevious = latest.sum(), previous.sum()
        if min(nLatest, nPrevious) < self.MIN_LINES_PER_WINDOW:
            return None

        change = np.median(phases[latest], axis=0) - np.median(phases[previous], axis=0)
        # Channels with no noise (e.g. the reference channel, whose relative phase is always 0) don't count
        use = self.noise > 1e-12
        if not np.any(use):
            return None
        spread = self.MEDIAN_SE_FACTOR * self.noise[use] * math.sqrt(1.0 / nLatest + 1.0 / nPrevious)
        z = change[use] / spread
        return float(np.sqrt(np.mean(z ** 2)))

# Methods by name
METHODS = {
    MedianFilter.name: MedianFilter,
}

def create_bulk_motion(method, trMs, params=None):
    """
    The bulk motion method called `method` (see METHODS), for a sequence with TR trMs (ms, or None), with
    its parameters from params (a dict, e.g. {'minWindowS': 5, 'minQuality': 0}).  method 'none' returns
    None (no bulk motion score).  Raises ValueError for an unknown method or parameter.
    """
    if method == 'none':
        return None
    if method not in METHODS:
        raise ValueError("Unknown bulk motion method %r (known: %s, or 'none')" % (method, ', '.join(sorted(METHODS))))
    cls = METHODS[method]
    params = dict(params or {})
    try:
        return cls(trMs, **params)
    except TypeError as e:
        raise ValueError("Bad parameters for bulk motion method %r: %s" % (method, e))
