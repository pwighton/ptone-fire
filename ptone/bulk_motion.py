# Real-time bulk head motion score from the pilot tone.
#
# A score of how much the head has moved from one position to another, computed line by line from the
# estimates pilottone.py already calculates (analyze_line()), using only lines already seen, so it can run
# live.
#
# Methods share one interface, so others can be added later.  Each is created with the scan's TR and
# reference channel, and its own parameters:
#   method = create_bulk_motion('medianFilter', trMs, refChanIdx, {'minWindowS': 3, 'minQuality': 0.5})
#   score = method.update(timeMs, relativePhase, relativeAmplitude, quality)    # once per analysed line
# update() returns a new score when it has one, else None.  Every method receives every line, with relative
# phase and amplitude for all channels and the line's quality, and decides itself which to use.

import math

import numpy as np

class MedianFilter:
    """
    Bulk motion score: the change in each channel's median relative phase between consecutive time windows.

    Relative phase behaves like a position: it depends on where the head is, so the head moving from one
    position to another shifts many channels' relative phase from one level to another.

    - The scan is divided into back-to-back windows of minWindowS seconds, starting at the first line used.
    - When a window is complete (a line arrives after its end), each channel's median relative phase over
      the window is taken, if the window has at least MIN_LINES_PER_WINDOW lines.  A median switches
      cleanly to a new level when the head moves, and ignores odd outlying lines.
    - Score: the change in each channel's median from the most recent earlier window that had enough lines,
      combined across channels as a root-mean-square, in radians.  The reference channel (refChanIdx),
      whose relative phase is always 0, doesn't count.  Comparing with the most recent window with data, rather than
      strictly the previous window, keeps scores coming across gaps with no lines (e.g. the 6 s between a
      TSE's bursts of lines).
    - One score per window with data, from the second such window on.  A movement raises the score of the
      window it ends in, and of the next one too if it happens in the second half of a window.

    The score isn't scaled: in typical sessions a still head scores ~0.003 rad and large movements (>= 2 mm
    or deg) mostly > ~0.015 rad (see the development notes).  Window length is a trade-off: shorter windows
    give earlier and more frequent scores but noisier medians; on the test data FLASH and SWI did best with
    3-5 s windows, the TSE equally well with 1-2 s.

    Parameters:
      trMs:       the sequence TR in ms (part of the shared interface; not used by this method)
      refChanIdx: the reference channel for relative phase and amplitude (pilottone.json refChanIdx)
      minWindowS: window length in seconds (default 3)
      minQuality: lines whose quality is below this are skipped (pilottone.py passes ptoneQualityThreshold)

    Relative amplitude is ignored.  Relative phase isn't unwrapped: it relies on pilottone.py setting the
    phase midpoint from a line with the tone (ptoneQualityThreshold), which keeps it continuous in practice.
    """

    name = 'medianFilter'

    MIN_LINES_PER_WINDOW = 5        # Windows with fewer lines have no median (e.g. during a gap)

    def __init__(self, trMs=None, refChanIdx=0, minWindowS=3.0, minQuality=0.0):
        minWindowS = float(minWindowS)
        if minWindowS <= 0:
            raise ValueError("minWindowS must be positive (got %s)" % minWindowS)
        self.trMs = trMs
        self.refChanIdx = int(refChanIdx)
        self.minWindowS = minWindowS
        self.minQuality = float(minQuality)
        self.windowMs = minWindowS * 1000.0

        self.numChan = None
        self.firstTimeMs = None                     # Start of the first window
        self.windowIndex = None                     # Index of the window being filled
        self.windowPhases = []                      # Relative phases of the lines in it
        self.prevMedian = None                      # Median of the most recent earlier window with enough lines
        self.prevWindow = None                      # (start ms, end ms) of that window
        # Details of the last score, e.g. for evaluation
        self.scoredWindow = None                    # (start ms, end ms) of the window the last score is for
        self.comparedWindow = None                  # (start ms, end ms) of the window it was compared with

    def window_bounds(self, index):
        start = self.firstTimeMs + index * self.windowMs
        return start, start + self.windowMs

    def update(self, timeMs, relativePhase, relativeAmplitude, quality):
        """Add one analysed line.  Returns a new score if this line completes a window with one, else None."""
        if quality < self.minQuality:
            return None
        phase = np.asarray(relativePhase, dtype=float)
        if self.numChan is None:
            self.numChan = phase.shape[0]
            if not 0 <= self.refChanIdx < self.numChan:
                raise ValueError("refChanIdx %d out of range for %d channels" % (self.refChanIdx, self.numChan))
        elif phase.shape[0] != self.numChan:
            return None
        timeMs = float(timeMs)
        if self.firstTimeMs is None:
            self.firstTimeMs = timeMs
            self.windowIndex = 0

        # A line after the end of the window being filled completes it (windows in between are empty)
        score = None
        index = int(math.floor((timeMs - self.firstTimeMs) / self.windowMs))
        if index > self.windowIndex:
            score = self.complete_window()
            self.windowIndex = index
        self.windowPhases.append(phase)
        return score

    def complete_window(self):
        bounds = self.window_bounds(self.windowIndex)
        phases, self.windowPhases = self.windowPhases, []
        if len(phases) < self.MIN_LINES_PER_WINDOW:
            return None                             # Too few lines: keep comparing with the last window with data
        median = np.median(np.stack(phases), axis=0)
        score = None
        if self.prevMedian is not None:
            # The reference channel (relative phase always 0) doesn't count
            change = np.delete(median - self.prevMedian, self.refChanIdx)
            score = float(np.sqrt(np.mean(change ** 2))) if len(change) else 0.0
            self.scoredWindow = bounds
            self.comparedWindow = self.prevWindow
        self.prevMedian = median
        self.prevWindow = bounds
        return score

# Methods by name
METHODS = {
    MedianFilter.name: MedianFilter,
}

def create_bulk_motion(method, trMs, refChanIdx, params=None):
    """
    The bulk motion method called `method` (see METHODS), for a scan with sequence TR trMs (ms, or None)
    and reference channel refChanIdx (the channel relative phase and amplitude are relative to), with the
    method's own parameters from params (a dict, e.g. {'minWindowS': 3, 'minQuality': 0.5}).  method 'none'
    returns None (no bulk motion score).  Raises ValueError for an unknown method or parameter.
    """
    if method == 'none':
        return None
    if method not in METHODS:
        raise ValueError("Unknown bulk motion method %r (known: %s, or 'none')" % (method, ', '.join(sorted(METHODS))))
    cls = METHODS[method]
    params = dict(params or {})
    try:
        return cls(trMs, refChanIdx, **params)
    except TypeError as e:
        raise ValueError("Bad parameters for bulk motion method %r: %s" % (method, e))
