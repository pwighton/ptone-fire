#!/usr/bin/env python3
# Evaluate a bulk motion method (ptone/bulk_motion.py) against TCL head tracking.
#
# Each pilottone.py results file (.npz) with TCL data (added by ptone_plot.py --add-tcl or
# ptone-offline-batch) is replayed line by line through the method, exactly as pilottone.py would run it
# live.  Any method in ptone/bulk_motion.py can be evaluated: each score says how much the head moved
# between two spans of time, which the method reports with the score (its scoredWindow and comparedWindow,
# see ptone/bulk_motion.py).  Each score is compared with how far the head moved between the same two
# spans according to the tracker: Jenkinson's RMS deviation (Jenkinson 1999, "Measuring transformation
# error by RMS deviation"; the RMS displacement over a sphere of radius RADIUS_MM) between the tracker's
# median pose over each span.  The tracker is shifted LAG_S earlier, since the tone changes ~0.4 s before
# the tracker reports a movement (probably tracker latency).
#
# The tracker (TracSuite) wasn't cross-calibrated to the scanner, so its poses (Tx, Ty, Tz in mm; Rx, Ry, Rz
# in degrees) are in its own coordinates.  The sphere is centred at CENTRE_MM in those coordinates (default
# its origin, i.e. about its rotation origin), and rotations are taken as R = Rx Ry Rz (the order isn't
# documented; it hardly matters for head-motion angles).  TracSuite's own '3D motion' isn't used: it's a
# displacement from the reference pose at the start of tracking, so its change between two windows can
# understate a movement, and its exact definition couldn't be confirmed from the data.
#
# Method parameters: the method's defaults, or with --config a pilottone.py JSON config (e.g. pilottone.json):
# each scan gets the method's parameters from it (<method><Parameter> settings, e.g. medianFilterMinWindowS)
# after applying its protocolOverrides rules to the scan's protocol name, exactly as pilottone.py does
# (ptone/protocol_overrides.py).  --param and --sweep values apply on top, to every scan.
#
# Results are grouped by sequence (from the protocol name: TSE, FLASH, SWI, MPRAGE, else the protocol
# name).  Context groups (default MPRAGE) are reported but left out of pooled results.  Scans are 'm'
# (subject moved on instruction) or 'nm' (asked to keep still) from '--m-' / '--nm-' in the protocol name.
#
# Outputs, in the output directory:
#   windows.csv   one row per score: the scan, the time spans compared, the score, the tracker's change
#   summary.csv   per group (and parameter value, with --sweep): first-score time, scores per scan, still-head
#                 level (still windows in 'nm' scans), AUC moved vs still (overall and by movement size),
#                 false alarms and detection at reading-guide thresholds, and a leave-one-session-out check
#   sessions.csv  still-head level per session and group (to spot unusual sessions)
#   plots/        with --plots: per scan, the score and the tracker's change over time
#
# Command line:
#   ptone-bulk-motion-eval <results dir or .npz> ... --out-dir eval [--config pilottone.json] [--param minWindowS=2]
#                          [--sweep minWindowS=1,2,3,5] [--exclude-sessions ptoneH20260429] [--plots]
#   (or python ptone/bulk_motion_eval.py ...)
# Python:
#   from ptone.bulk_motion_eval import evaluate; windows = evaluate(['results/'], params={'minWindowS': 2})

import argparse
import glob
import html
import json
import os
import re
import sys

import numpy as np

try:
    from ptone.bulk_motion import create_bulk_motion, config_params
except ImportError:                 # Run as a script from ptone/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ptone.bulk_motion import create_bulk_motion, config_params
from ptone.protocol_overrides import apply_protocol_overrides

LAG_S = 0.4                         # Tracker shift: the tone changes ~0.4 s before the tracker reports a movement
RADIUS_MM = 60.0                    # Jenkinson's RMS deviation: sphere radius (mm), about the size of a head
CENTRE_MM = (0.0, 0.0, 0.0)         # ... and centre, in the tracker's coordinates
MOVED = 0.5                         # Moved: RMS deviation between the windows >= this (mm)
STILL = 0.2                         # Still: RMS deviation < this (mm)
SIZE_BINS = [(0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]   # Movement sizes (RMS deviation, mm)
THRESHOLDS = [0.01, 0.015, 0.02]    # Reading-guide score thresholds (rad) to report
TARGET_FA = 0.02                    # False-alarm rate the leave-one-session-out threshold aims for
CONTEXT_GROUPS = ['MPRAGE']         # Reported, but left out of pooled results
GROUP_PATTERNS = [('TSE', r'tse'), ('FLASH', r'fl2d|flash'), ('SWI', r'swi'), ('MPRAGE', r'mprage')]

# ----- Reading results ---------------------------------------------------------------------------

def find_results(paths):
    """.npz files from a list of files and directories (searched recursively), sorted."""
    found = []
    for p in paths:
        if os.path.isdir(p):
            found += glob.glob(os.path.join(p, '**', '*.npz'), recursive=True)
        else:
            found.append(p)
    return sorted(set(found))

def protocol_name(d):
    # The protocol name from the saved MRD header, or '' if it can't be read
    m = re.search(r'<protocolName>(.*?)</protocolName>', str(d['mrd_header'])) if 'mrd_header' in d.files else None
    return html.unescape(m.group(1)) if m else ''

def sequence_group(protocol):
    """The sequence group for a protocol name (see GROUP_PATTERNS), else the protocol without its motion suffix."""
    for group, pattern in GROUP_PATTERNS:
        if re.search(pattern, protocol, re.IGNORECASE):
            return group
    return re.sub(r'--n?m(-n?pt)?$', '', protocol)

def motion_kind(protocol):
    """'m' (moved on instruction), 'nm' (asked to keep still) or '?' from the protocol name."""
    if re.search(r'--m(-|$)', protocol):
        return 'm'
    if re.search(r'--nm(-|$)', protocol):
        return 'nm'
    return '?'

def tr_ms(d):
    # The sequence TR from the saved MRD header, or None
    m = re.search(r'<TR>([0-9.eE+-]+)</TR>', str(d['mrd_header'])) if 'mrd_header' in d.files else None
    return float(m.group(1)) if m else None

def load_scan(npzPath):
    """The arrays and labels evaluate() needs from a results file, or None if it has no TCL data."""
    with np.load(npzPath) as d:
        if 'tcl_Tx' not in d.files:
            return None
        settings = json.loads(str(d['settings'])) if 'settings' in d.files else {}
        protocol = protocol_name(d)
        name = os.path.basename(npzPath)
        mid = re.search(r'MID(\d+)', name)
        return dict(
            path=npzPath, session=name.split('--')[0], mid=int(mid.group(1)) if mid else None,
            protocol=protocol, group=sequence_group(protocol), kind=motion_kind(protocol), trMs=tr_ms(d),
            refChanIdx=int(settings.get('refChanIdx', 0)),
            qualityThreshold=float(settings.get('ptoneQualityThreshold', 0.5)),
            timeMs=d['time_ms'].astype(float), quality=d['quality'].astype(float),
            relativePhase=d['relative_phase'].astype(float), relativeAmplitude=d['relative_amplitude'].astype(float),
            position=np.stack([d['tcl_Tx'], d['tcl_Ty'], d['tcl_Tz']], 1).astype(float),
            rotation=np.stack([d['tcl_Rx'], d['tcl_Ry'], d['tcl_Rz']], 1).astype(float))

# ----- Scoring and the tracker -------------------------------------------------------------------

def rotation_matrix(rotationDeg):
    """Rotation matrix R = Rx Ry Rz from rotations about x, y and z in degrees."""
    a, b, c = np.radians(rotationDeg)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rx @ ry @ rz

def jenkinson_rms(translationA, rotationA, translationB, rotationB, radius=RADIUS_MM, centre=CENTRE_MM):
    """
    Jenkinson's RMS deviation (mm) between two poses (translation in mm, rotation in degrees): the RMS
    displacement of points in a sphere of the given radius and centre when moving from pose A to pose B.
    RMS^2 = (radius^2 / 5) tr(A^T A) + |t + A c|^2, for the relative transform x -> R x + t and A = R - I.
    """
    ra, rb = rotation_matrix(rotationA), rotation_matrix(rotationB)
    r = rb @ ra.T                                               # Relative rotation: pose A -> pose B
    t = np.asarray(translationB, float) - r @ np.asarray(translationA, float)
    a = r - np.eye(3)
    v = t + a @ np.asarray(centre, float)
    return float(np.sqrt(radius ** 2 / 5.0 * np.trace(a.T @ a) + v @ v))

def score_scan(scan, method='medianFilter', params=None, lagS=LAG_S, minQuality=None, radius=RADIUS_MM,
               centre=CENTRE_MM):
    """
    Replay a scan through the method; returns one dict per score: the two time spans the score compares (the
    method's scoredWindow and comparedWindow, as columns windowStartS/EndS and comparedStartS/EndS), the
    score, and how far the head moved between the same spans according to the tracker: Jenkinson's RMS
    deviation (trackerRmsMm) between the tracker's median pose over each span, plus the change in
    translation (trackerMm) and the largest change in rotation (trackerDeg), from lines with the tone
    (quality >= minQuality), with the tracker shifted lagS earlier.  minQuality: default the file's
    ptoneQualityThreshold.
    """
    minQuality = scan['qualityThreshold'] if minQuality is None else minQuality
    params = dict(params or {})
    params.setdefault('minQuality', minQuality)
    m = create_bulk_motion(method, scan['trMs'], scan['refChanIdx'], params)
    t, q = scan['timeMs'], scan['quality']
    good = q >= minQuality
    tTracker = t - lagS * 1000
    firstMs = t[good][0] if good.any() else t[0]
    rows = []
    for i in range(len(t)):
        s = m.update(t[i], scan['relativePhase'][i], scan['relativeAmplitude'][i], q[i])
        if s is None:
            continue
        if getattr(m, 'scoredWindow', None) is None or getattr(m, 'comparedWindow', None) is None:
            raise ValueError("Bulk motion method %r didn't report the time spans its score compares "
                             "(scoredWindow and comparedWindow)" % method)
        (a, b), (pa, pb) = m.scoredWindow, m.comparedWindow
        cur = good & (tTracker >= a) & (tTracker < b)
        prev = good & (tTracker >= pa) & (tTracker < pb)
        if cur.any() and prev.any():
            posCur, posPrev = np.median(scan['position'][cur], 0), np.median(scan['position'][prev], 0)
            rotCur, rotPrev = np.median(scan['rotation'][cur], 0), np.median(scan['rotation'][prev], 0)
            rms = jenkinson_rms(posPrev, rotPrev, posCur, rotCur, radius, centre)
            dT = float(np.linalg.norm(posCur - posPrev))
            dR = float(np.max(np.abs(rotCur - rotPrev)))
        else:
            rms = dT = dR = np.nan
        rows.append(dict(session=scan['session'], mid=scan['mid'], protocol=scan['protocol'], group=scan['group'],
                         kind=scan['kind'], file=os.path.basename(scan['path']),
                         windowStartS=(a - firstMs) / 1000, windowEndS=(b - firstMs) / 1000,
                         comparedStartS=(pa - firstMs) / 1000, comparedEndS=(pb - firstMs) / 1000,
                         score=s, trackerRmsMm=rms, trackerMm=dT, trackerDeg=dR))
    return rows

def evaluate(paths, method='medianFilter', params=None, lagS=LAG_S, minQuality=None, moved=MOVED, still=STILL,
             excludeSessions=(), radius=RADIUS_MM, centre=CENTRE_MM, config=None):
    """
    Score every results file with TCL data under paths, except those from sessions in excludeSessions
    (session = the start of the filename, e.g. ptoneH20260429).  Returns (windows, scans): one dict per score,
    with 'size' (the RMS deviation, mm), 'moved' (size >= moved) and 'still' (size < still) added; and one
    dict per scan with its first-score time and number of scores.  Both include the protocolOverrides rule
    used ('protocolOverride', the pattern, '' if none) and the method parameters used ('scanParams').

    config: a pilottone.py JSON config ({'parameters': {...}}), or None.  If given, each scan's method
    parameters come from it, after applying its protocolOverrides rules to the scan's protocol name; params
    apply on top.
    """
    windows, scans = [], []
    for path in find_results(paths):
        scan = load_scan(path)
        if scan is None or scan['session'] in excludeSessions:
            continue
        scanParams, override = dict(params or {}), None
        if config is not None:
            scanConfig, applied = apply_protocol_overrides(config, scan['protocol'])
            scanParams = dict(config_params(scanConfig, method) or {}, **scanParams)
            override = applied['protocolOverrideMatch']
        rows = score_scan(scan, method, scanParams, lagS, minQuality, radius, centre)
        for r in rows:
            r['protocolOverride'] = override or ''
            r['scanParams'] = json.dumps(scanParams, sort_keys=True)
            r['size'] = r['trackerRmsMm']
            r['moved'] = bool(r['size'] >= moved)
            r['still'] = bool(r['size'] < still)
        windows += rows
        scans.append(dict(session=scan['session'], mid=scan['mid'], group=scan['group'], kind=scan['kind'],
                          firstScoreS=rows[0]['windowEndS'] if rows else np.nan, numScores=len(rows),
                          protocolOverride=override or '', scanParams=json.dumps(scanParams, sort_keys=True)))
    return windows, scans

# ----- Summaries ---------------------------------------------------------------------------------

def auc(positive, negative):
    """Probability that a random positive scores higher than a random negative (ties count half); nan if either is empty."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    values = np.r_[positive, negative]
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for v in np.unique(values):                     # Average ranks for ties
        tie = values == v
        ranks[tie] = ranks[tie].mean()
    return (ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))

def leave_one_session_out(windows, targetFa=TARGET_FA):
    """
    For each session: the threshold that gives targetFa false alarms on the still windows of 'nm' scans in
    the other sessions, applied to this session.  Returns (median held-out false-alarm rate, worst, and
    {movement size lower bound: fraction of movements >= it caught}).
    """
    sessions = sorted(set(w['session'] for w in windows))
    fas, caught = [], {lo: [0, 0] for lo, _ in SIZE_BINS}
    for s in sessions:
        train = [w['score'] for w in windows if w['session'] != s and w['still'] and w['kind'] == 'nm']
        test = [w for w in windows if w['session'] == s]
        if len(train) < 10 or not test:
            continue
        thr = np.quantile(train, 1 - targetFa)
        stillTest = [w['score'] for w in test if w['still'] and w['kind'] == 'nm']
        if stillTest:
            fas.append(np.mean(np.array(stillTest) > thr))
        for lo, _ in SIZE_BINS:
            mv = [w['score'] for w in test if w['size'] >= lo]
            caught[lo][0] += int(np.sum(np.array(mv) > thr)); caught[lo][1] += len(mv)
    if not fas:
        return np.nan, np.nan, {}
    return float(np.median(fas)), float(np.max(fas)), {lo: (c / n if n else np.nan) for lo, (c, n) in caught.items()}

def summarize(windows, scans, thresholds=THRESHOLDS, targetFa=TARGET_FA, contextGroups=CONTEXT_GROUPS):
    """One summary dict per group, plus 'pooled' (all groups except contextGroups)."""
    groups = sorted(set(s['group'] for s in scans))
    out = []
    for group in groups + ['pooled']:
        if group == 'pooled':
            ws = [w for w in windows if w['group'] not in contextGroups]
            ss = [s for s in scans if s['group'] not in contextGroups]
        else:
            ws = [w for w in windows if w['group'] == group]
            ss = [s for s in scans if s['group'] == group]
        score = np.array([w['score'] for w in ws]); size = np.array([w['size'] for w in ws])
        moved = np.array([w['moved'] for w in ws], bool); still = np.array([w['still'] for w in ws], bool)
        stillNm = still & np.array([w['kind'] == 'nm' for w in ws], bool)
        row = dict(group=group, context=group in contextGroups, scans=len(ss),
                   scansWithoutScores=sum(1 for s in ss if s['numScores'] == 0),
                   firstScoreS=float(np.nanmedian([s['firstScoreS'] for s in ss])) if ss else np.nan,
                   scoresPerScan=float(np.median([s['numScores'] for s in ss])) if ss else np.nan,
                   stillWindows=int(stillNm.sum()), movedWindows=int(moved.sum()),
                   stillMedian=float(np.median(score[stillNm])) if stillNm.any() else np.nan,
                   stillP90=float(np.percentile(score[stillNm], 90)) if stillNm.any() else np.nan,
                   stillP99=float(np.percentile(score[stillNm], 99)) if stillNm.any() else np.nan,
                   aucMoved=auc(score[moved], score[still]))
        for lo, hi in SIZE_BINS:
            sel = (size >= lo) & (size < hi)
            row['auc_%g_%g' % (lo, hi)] = auc(score[sel], score[still])
        for thr in thresholds:
            row['fa_%g' % thr] = float(np.mean(score[stillNm] > thr)) if stillNm.any() else np.nan
            for lo, _ in SIZE_BINS:
                sel = size >= lo
                row['caught_%g_ge%g' % (thr, lo)] = float(np.mean(score[sel] > thr)) if sel.any() else np.nan
        faMedian, faWorst, caught = leave_one_session_out(ws, targetFa)
        row['losoFaMedian'], row['losoFaWorst'] = faMedian, faWorst
        for lo, v in caught.items():
            row['losoCaught_ge%g' % lo] = v
        out.append(row)
    return out

def session_levels(windows):
    """Still-head score level per session and group (still windows in 'nm' scans)."""
    out = []
    for session in sorted(set(w['session'] for w in windows)):
        for group in sorted(set(w['group'] for w in windows if w['session'] == session)):
            v = [w['score'] for w in windows if w['session'] == session and w['group'] == group and w['still'] and w['kind'] == 'nm']
            if v:
                out.append(dict(session=session, group=group, stillWindows=len(v), stillMedian=float(np.median(v)),
                                stillP90=float(np.percentile(v, 90))))
    return out

# ----- Output ------------------------------------------------------------------------------------

def write_csv(rows, path):
    import pandas
    pandas.DataFrame(rows).to_csv(path, index=False)

def plot_scans(windows, plotDir):
    # Per scan: score and the tracker's change over time
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    os.makedirs(plotDir, exist_ok=True)
    for f in sorted(set(w['file'] for w in windows)):
        ws = [w for w in windows if w['file'] == f]
        t = [w['windowEndS'] for w in ws]
        fig = Figure(figsize=(10, 4)); FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.plot(t, [w['score'] for w in ws], 'o-', color='tab:blue', label='score (rad)')
        ax.set_xlabel('Window end (s from the first line with the tone)'); ax.set_ylabel('score (rad)', color='tab:blue')
        ax2 = ax.twinx()
        ax2.plot(t, [w['size'] for w in ws], 's--', color='tab:red', label='tracker RMS deviation (mm)')
        ax2.set_ylabel('tracker RMS deviation between the windows (mm)', color='tab:red')
        ax.set_title('%s (%s, %s)' % (f, ws[0]['group'], ws[0]['kind']), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(plotDir, os.path.splitext(f)[0] + '--bulk-motion.png'), dpi=80)

def format_summary(summary, label=''):
    def pct(v, digits=0):
        return 'n/a' if v is None or not np.isfinite(v) else '%.*f%%' % (digits, 100 * v)
    lines = []
    for r in summary:
        name = r['group'] + (' (context)' if r['context'] else '')
        if np.isfinite(r['losoFaMedian']):
            loso = ("FA median %s, worst %s, catches %s of movements >= %s mm"
                    % (pct(r['losoFaMedian'], 1), pct(r['losoFaWorst'], 1),
                       ' / '.join(pct(r.get('losoCaught_ge%g' % lo, np.nan)) for lo, _ in SIZE_BINS),
                       ' / '.join('%g' % lo for lo, _ in SIZE_BINS)))
        else:
            loso = 'n/a (needs at least 2 sessions)'
        lines.append("%s%-16s scans %3d (%d without scores) | first score %5.1f s, %4.0f per scan | still: median %.4f, 99th pct %.4f rad"
                     " | AUC moved %.2f (>=2 mm %.2f) | leave-one-session-out: %s"
                     % (label, name, r['scans'], r['scansWithoutScores'], r['firstScoreS'], r['scoresPerScan'],
                        r['stillMedian'], r['stillP99'], r['aucMoved'], r.get('auc_2_inf', np.nan), loso))
    return '\n'.join(lines)

def parse_values(text):
    # 'name=value' -> (name, value); values stay strings (the method converts them)
    if '=' not in text:
        raise argparse.ArgumentTypeError("expected name=value, got %r" % text)
    name, value = text.split('=', 1)
    return name.strip(), value.strip()

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a bulk motion method against TCL head tracking")
    parser.add_argument('inputs', nargs='+', help='Results files (.npz) or directories (searched recursively)')
    parser.add_argument('--out-dir', required=True, help='Directory for windows.csv, summary.csv, sessions.csv and plots')
    parser.add_argument('--method', default='medianFilter', help='Bulk motion method (default: medianFilter)')
    parser.add_argument('--config', default=None, metavar='JSON',
                        help="pilottone.py config (e.g. pilottone.json): each scan's method parameters from it, after its "
                             "protocolOverrides rules (default: the method's defaults)")
    parser.add_argument('--param', type=parse_values, action='append', default=[], metavar='NAME=VALUE',
                        help='Method parameter, e.g. minWindowS=2 (repeatable)')
    parser.add_argument('--sweep', type=parse_values, default=None, metavar='NAME=V1,V2,...',
                        help='Evaluate each value of one parameter, e.g. minWindowS=1,2,3,5')
    parser.add_argument('--min-quality', type=float, default=None,
                        help="Lines used (default: each file's ptoneQualityThreshold)")
    parser.add_argument('--lag', type=float, default=LAG_S, help='Tracker shift in seconds (default %g)' % LAG_S)
    parser.add_argument('--moved', type=float, default=MOVED, help='Moved: tracker RMS deviation >= this, mm (default %g)' % MOVED)
    parser.add_argument('--still', type=float, default=STILL, help='Still: tracker RMS deviation < this, mm (default %g)' % STILL)
    parser.add_argument('--radius', type=float, default=RADIUS_MM,
                        help="Jenkinson's RMS deviation: sphere radius in mm (default %g)" % RADIUS_MM)
    parser.add_argument('--centre', type=float, nargs=3, default=list(CENTRE_MM), metavar=('X', 'Y', 'Z'),
                        help="... and centre in the tracker's coordinates, mm (default 0 0 0)")
    parser.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS, help='Reading-guide thresholds (rad)')
    parser.add_argument('--target-fa', type=float, default=TARGET_FA, help='Leave-one-session-out false-alarm target (default %g)' % TARGET_FA)
    parser.add_argument('--context-groups', nargs='*', default=CONTEXT_GROUPS, help='Groups left out of pooled results (default: MPRAGE)')
    parser.add_argument('--exclude-sessions', nargs='*', default=[], metavar='SESSION',
                        help='Sessions to leave out, e.g. ptoneH20260429')
    parser.add_argument('--plots', action='store_true', help='Plot each scan (score and tracker change)')
    args = parser.parse_args(argv)

    config = None
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
    params = dict(args.param)
    runs = [(None, params)]
    if args.sweep:
        name, values = args.sweep
        runs = [('%s=%s' % (name, v), dict(params, **{name: v})) for v in values.split(',')]

    os.makedirs(args.out_dir, exist_ok=True)
    allWindows, allSummary, allSessions = [], [], []
    for label, p in runs:
        windows, scans = evaluate(args.inputs, args.method, p, args.lag, args.min_quality, args.moved, args.still,
                                  args.exclude_sessions, args.radius, args.centre, config)
        if not windows:
            print("bulk_motion_eval: no scores (no results with TCL data under %s?)" % ', '.join(args.inputs), file=sys.stderr)
            return 1
        summary = summarize(windows, scans, args.thresholds, args.target_fa, args.context_groups)
        sessions = session_levels(windows)
        for rows in (windows, summary, sessions):
            for r in rows:
                r['method'] = args.method
                r['params'] = json.dumps(p, sort_keys=True)
        allWindows += windows; allSummary += summary; allSessions += sessions
        print(format_summary(summary, '' if label is None else '[%s] ' % label))
        if args.plots:
            plot_scans(windows, os.path.join(args.out_dir, 'plots' if label is None else 'plots--' + label.replace('=', '-')))
    write_csv(allWindows, os.path.join(args.out_dir, 'windows.csv'))
    write_csv(allSummary, os.path.join(args.out_dir, 'summary.csv'))
    write_csv(allSessions, os.path.join(args.out_dir, 'sessions.csv'))
    print("bulk_motion_eval: wrote windows.csv, summary.csv and sessions.csv to %s" % args.out_dir)
    return 0

if __name__ == '__main__':
    sys.exit(main())
