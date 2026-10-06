# Tests for ptone/bulk_motion.py
#
# The real-data test looks for the re-run results in $PTONE_TEST_DATA (default ../pilot-tone-test-data
# relative to the repository root) and is skipped if they're not found.

import glob
import json
import os

import numpy as np
import pytest

from ptone.bulk_motion import MedianFilter, create_bulk_motion

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataDir = os.environ.get('PTONE_TEST_DATA', os.path.join(repoDir, '..', 'pilot-tone-test-data'))

def lines(durationS=60.0, spacingMs=10.0, numChan=8, noise=0.003, seed=0, gaps=None):
    # Times (ms) and relative phases [line, channel] for a still head: noise around a fixed level per
    # channel; channel 0 is the reference (always 0).  gaps: list of (start s, end s) with no lines
    rng = np.random.default_rng(seed)
    t = np.arange(0, durationS * 1000, spacingMs)
    for a, b in (gaps or []):
        t = t[(t < a * 1000) | (t >= b * 1000)]
    ph = rng.uniform(-1, 1, numChan) + noise * rng.standard_normal((len(t), numChan))
    ph[:, 0] = 0.0
    return t, ph

def run(method, t, ph, quality=None):
    # Scores from update(), as (index of the line that returned it, score)
    quality = np.ones(len(t)) if quality is None else quality
    out = []
    for i in range(len(t)):
        s = method.update(t[i], ph[i], None, quality[i])
        if s is not None:
            out.append((i, s))
    return out

# ----- Creating methods --------------------------------------------------------------------------

def test_create():
    m = create_bulk_motion('medianFilter', 1900.0, 0, {'minWindowS': 2, 'minQuality': 0.5})
    assert isinstance(m, MedianFilter) and m.windowMs == 2000 and m.minQuality == 0.5
    assert create_bulk_motion('medianFilter', None, 0).windowMs == 3000          # Default 3 s
    assert create_bulk_motion('none', 1900.0, 0) is None
    with pytest.raises(ValueError, match='Unknown bulk motion method'):
        create_bulk_motion('meanFilter', 1900.0, 0)
    with pytest.raises(ValueError, match='Bad parameters'):
        create_bulk_motion('medianFilter', 1900.0, 0, {'windowS': 2})
    with pytest.raises(ValueError, match='minWindowS must be positive'):
        create_bulk_motion('medianFilter', 1900.0, 0, {'minWindowS': 0})
    with pytest.raises(ValueError, match='minLinesPerWindow must be at least 1'):
        create_bulk_motion('medianFilter', 1900.0, 0, {'minLinesPerWindow': 0})

def test_parameters_and_defaults():
    # The parameters pilottone.json can set (as medianFilter<Parameter>), with defaults matching the constructor
    assert MedianFilter.PARAMETERS == {'minWindowS': 3.0, 'minLinesPerWindow': 5, 'maxGapS': None}
    m = create_bulk_motion('medianFilter', None, 0)
    assert (m.minWindowS, m.minLinesPerWindow, m.maxGapS) == (3.0, 5, None)
    # maxGapS <= 0 means no limit
    assert create_bulk_motion('medianFilter', None, 0, {'maxGapS': 0}).maxGapS is None
    assert create_bulk_motion('medianFilter', None, 0, {'maxGapS': '7.5'}).maxGapS == 7.5

# ----- Windows and scores ------------------------------------------------------------------------

def test_one_score_per_window_from_the_second():
    t, ph = lines(durationS=20.0)
    m = create_bulk_motion('medianFilter', None, 0, {'minWindowS': 3})
    scores = run(m, t, ph)
    # Windows end at 3, 6, ..., 18 s; the first complete window (0-3 s) gives no score, and the last
    # (18-20 s) is never completed.  Each score comes on the first line after its window's end
    assert [t[i] for i, _ in scores] == [6000, 9000, 12000, 15000, 18000]
    assert m.scoredWindow == (15000, 18000) and m.comparedWindow == (12000, 15000)

def test_still_head_scores_small():
    t, ph = lines(durationS=60.0, noise=0.003)
    scores = np.array([s for _, s in run(create_bulk_motion('medianFilter', None, 0), t, ph)])
    # Window medians of 300 lines of 0.003 rad noise differ by ~0.0003 rad
    assert np.all(scores < 0.001)

def test_step_raises_the_score_of_its_window():
    t, ph = lines(durationS=30.0)
    # 4 of 7 channels shift at 13 s, a third of the way into the 12-15 s window: more than half of that
    # window is after the step, so its median moves to the new level
    ph[t >= 13000, 1:5] += 0.05
    m = create_bulk_motion('medianFilter', None, 0, {'minWindowS': 3})
    scores = run(m, t, ph)
    byWindowEnd = {t[i]: s for i, s in scores}        # Keyed by the end of the window each score is for
    expected = np.sqrt(4 * 0.05 ** 2 / 7)             # RMS over the 7 non-reference channels
    assert byWindowEnd[15000] == pytest.approx(expected, rel=0.1)
    # Next window: nearly the same level (the step window's median is slightly short of the new level,
    # since a third of its lines are before the step)
    assert byWindowEnd[18000] < 0.005
    assert max(s for w, s in byWindowEnd.items() if w <= 12000) < 0.001

def test_step_late_in_a_window_shows_in_the_next():
    t, ph = lines(durationS=30.0)
    ph[t >= 14800, 1:5] += 0.05                       # In the last 7% of the 12-15 s window: its median stays
    scores = {t[i]: s for i, s in run(create_bulk_motion('medianFilter', None, 0, {'minWindowS': 3}), t, ph)}
    assert scores[15000] < 0.01 and scores[18000] > 0.03

def test_periodic_change_within_a_window_averages_out():
    # A 1 s oscillation (e.g. a heartbeat) doesn't change 3 s window medians
    t, ph = lines(durationS=40.0)
    ph[:, 1:] += 0.03 * np.sin(2 * np.pi * t / 1000.0)[:, None]
    scores = np.array([s for _, s in run(create_bulk_motion('medianFilter', None, 0, {'minWindowS': 3}), t, ph)])
    assert np.all(scores < 0.003)

def test_reference_channel_not_counted():
    t, ph = lines(durationS=20.0, numChan=2)
    ph[t >= 6000, 1] += 0.04                          # At a window boundary, so the 6-9 s window is all new
    scores = {t[i]: s for i, s in run(create_bulk_motion('medianFilter', None, 0, {'minWindowS': 3}), t, ph)}
    assert scores[9000] == pytest.approx(0.04, rel=0.05)      # Only channel 1, not RMS with the zero channel

def test_reference_channel_from_parameter():
    # Reference channel 2: channels 0 and 1 count, channel 2 doesn't
    t, ph = lines(durationS=20.0, numChan=3)
    ph[:, 0] = 0.5 + 0.003 * np.random.default_rng(2).standard_normal(len(t))
    ph[:, 2] = 0.0
    ph[t >= 6000, 0] += 0.04
    m = create_bulk_motion('medianFilter', None, 2, {'minWindowS': 3})
    scores = {t[i]: s for i, s in run(m, t, ph)}
    assert scores[9000] == pytest.approx(np.sqrt(0.04 ** 2 / 2), rel=0.1)   # RMS over channels 0 and 1

def test_reference_channel_out_of_range():
    t, ph = lines(durationS=5.0, numChan=4)
    m = create_bulk_motion('medianFilter', None, 4)
    with pytest.raises(ValueError, match='refChanIdx 4 out of range'):
        m.update(t[0], ph[0], None, 1.0)

# ----- Gaps (e.g. a TSE: 3 s of lines every 9 s) -------------------------------------------------

def test_gaps_compare_with_the_last_window_with_data():
    gaps = [(3 + 9 * k, 9 + 9 * k) for k in range(6)]          # Lines only in 0-3, 9-12, 18-21, ... s
    t, ph = lines(durationS=54.0, gaps=gaps)
    ph[t >= 24000, 1:] += 0.05                                 # Move during the gap between 21 and 27 s
    m = create_bulk_motion('medianFilter', 9000.0, 0, {'minWindowS': 1})
    scores = run(m, t, ph)
    # Scores only come during bursts (1 s windows), first at 2 s
    assert t[scores[0][0]] == 2000
    byEnd = {m_end: s for m_end, s in [(t[i], s) for i, s in scores]}
    # The first window of the burst after the move (27-28 s) is compared with the last window before the
    # gap (20-21 s) and shows the move; neighbouring windows don't
    big = [(i, s) for i, s in scores if s > 0.02]
    assert len(big) == 1 and t[big[0][0]] == 28000
    assert max(s for i, s in scores if i != big[0][0]) < 0.002

def test_window_with_too_few_lines_is_skipped():
    t, ph = lines(durationS=12.0, spacingMs=10.0)
    keep = (t < 3000) | (t >= 6000) | (np.arange(len(t)) % 100 == 0)   # 3-6 s: only 3 lines
    t, ph = t[keep], ph[keep]
    m = create_bulk_motion('medianFilter', None, 0, {'minWindowS': 3})
    scores = run(m, t, ph)
    # 3-6 s has too few lines: 6-9 s is compared with 0-3 s
    assert [t[i] for i, _ in scores] == [9000]
    assert m.comparedWindow == (0, 3000)

def test_min_lines_per_window_parameter():
    t, ph = lines(durationS=12.0, spacingMs=10.0)
    keep = (t < 3000) | (t >= 6000) | (np.arange(len(t)) % 100 == 0)   # 3-6 s: only 3 lines
    t, ph = t[keep], ph[keep]
    m = create_bulk_motion('medianFilter', None, 0, {'minWindowS': 3, 'minLinesPerWindow': 3})
    scores = run(m, t, ph)
    # With a minimum of 3 lines, the 3-6 s window counts
    assert [t[i] for i, _ in scores] == [6000, 9000]

@pytest.mark.parametrize('maxGapS, bridged', [(None, True), (10, True), (5, False)])
def test_max_gap(maxGapS, bridged):
    # TSE-like: lines only in 0-3, 9-12 and 18-21 s; 1 s windows, so a 6 s gap between bursts
    gaps = [(3, 9), (12, 18)]
    t, ph = lines(durationS=21.0, gaps=gaps)
    m = create_bulk_motion('medianFilter', 9000.0, 0, {'minWindowS': 1, 'maxGapS': maxGapS})
    scoreTimes = [t[i] for i, _ in run(m, t, ph)]
    # Within a burst: windows ending at 2, 3 (scored on the line at 9 s, the next line), ...
    firstInSecondBurst = 10000.0                     # The 9-10 s window, scored on the line at 10 s
    assert (firstInSecondBurst in scoreTimes) == bridged
    assert 11000.0 in scoreTimes                     # Within the burst: always scored

# ----- Lines the method skips ---------------------------------------------------------------------

def test_low_quality_lines_skipped():
    t, ph = lines(durationS=30.0)
    rng = np.random.default_rng(1)
    bad = rng.random(len(t)) < 0.2
    quality = np.where(bad, 0.05, 0.97)
    garbage = ph.copy()
    garbage[bad, 1:] = rng.uniform(-np.pi, np.pi, (bad.sum(), ph.shape[1] - 1))   # Random phase without the tone
    withBad = [s for _, s in run(create_bulk_motion('medianFilter', None, 0, {'minQuality': 0.5}), t, garbage, quality)]
    goodOnly = [s for _, s in run(create_bulk_motion('medianFilter', None, 0, {'minQuality': 0.5}), t[~bad], ph[~bad])]
    assert withBad == pytest.approx(goodOnly)

def test_line_with_different_channel_count_ignored():
    t, ph = lines(durationS=12.0)
    m = create_bulk_motion('medianFilter', None, 0)
    assert m.update(t[0], ph[0], None, 1.0) is None
    assert m.update(t[1], ph[1][:4], None, 1.0) is None
    assert len(run(m, t[2:], ph[2:])) == 2             # Windows ending at 6 and 9 s (9-12 s never completes)

# ----- Real data ---------------------------------------------------------------------------------

def test_real_tse_largest_score_at_largest_tracked_movement():
    # The April motion TSE: the highest score falls on (or next to) the window with the largest tracked
    # change in head position
    paths = glob.glob(os.path.join(dataDir, '*', 'ptoneH20260429', '*MID00284*.npz'))
    if not paths:
        pytest.skip('Re-run results for ptoneH20260429 MID00284 not found in ' + dataDir)
    d = np.load(paths[0])
    if 'tcl_Tx' not in d.files:
        pytest.skip('No TCL data in ' + paths[0])
    t, q, ph = d['time_ms'].astype(float), d['quality'], d['relative_phase']
    pos = np.stack([d['tcl_Tx'], d['tcl_Ty'], d['tcl_Tz']], 1)
    refChanIdx = json.loads(str(d['settings']))['refChanIdx']
    m = create_bulk_motion('medianFilter', None, refChanIdx, {'minWindowS': 2, 'minQuality': 0.5})
    windows = []
    for i in range(len(t)):
        s = m.update(t[i], ph[i], None, q[i])
        if s is not None:
            (a, b), (pa, pb) = m.scoredWindow, m.comparedWindow
            cur = (t >= a) & (t < b) & (q >= 0.5); prev = (t >= pa) & (t < pb) & (q >= 0.5)
            windows.append((s, np.linalg.norm(np.median(pos[cur], 0) - np.median(pos[prev], 0))))
    scores = np.array([w[0] for w in windows]); moved = np.array([w[1] for w in windows])
    assert abs(int(np.argmax(scores)) - int(np.argmax(moved))) <= 1
    assert moved[np.argmax(scores)] > 1.0 or moved[max(np.argmax(scores) - 1, 0)] > 1.0
