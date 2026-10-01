#!/usr/bin/env python3
# Plot the pilot tone results saved by pilottone.py: for each channel, amplitude and phase against the
# scanner's line number, with the quality metric and optionally TCL head motion overlaid.
#
# Modelled on kstream's kstream/fast_phase_inspect.py (gen_plot).
#
# Command line:
#   python ptone/ptone_plot.py results.npz [--tcl MOTFILE] [--channel N ...] [--plot both|amplitude|phase] ...
# Python:
#   from ptone.ptone_plot import plot_npz
#   plot_npz('results.npz', tclMotPath='/path/to/2026-04-29_ptoneH20260429_120616_MOT.tsm')

import argparse
import json
import os
import re
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

def set_ylim_to_data(ax, values, good, constantHalfRange):
    # Y-limits around the good-quality values: their range plus a margin of 0.25 standard deviations.
    # If they're constant (e.g. the reference channel) or there are none, centre the line in a range of
    # +/- constantHalfRange(centre)
    v = values[good]
    v = v[np.isfinite(v)]
    if v.size and np.std(v) > 0:
        margin = 0.25 * np.std(v)
        ax.set_ylim(np.min(v) - margin, np.max(v) + margin)
    else:
        finite = values[np.isfinite(values)]
        centre = np.median(v) if v.size else (np.median(finite) if finite.size else 0.0)
        halfRange = constantHalfRange(centre)
        ax.set_ylim(centre - halfRange, centre + halfRange)

def short_git_commit(gitCommit):
    # First 7 characters of the commit ID, keeping anything after it, e.g. ' (with uncommitted changes)'
    return re.sub(r'^([0-9a-f]{7})[0-9a-f]+', r'\1', gitCommit)

def load_tcl_module():
    # ptone/tcl.py, whether this is imported as ptone.ptone_plot or run as a script from ptone/
    try:
        from ptone import tcl
    except ImportError:
        import tcl
    return tcl

def plot_npz(npzPath, pngPath=None, tclMotPath=None, channels=None, plot='both', amplitude='relative',
             showQuality=True, qualityThreshold=None, legendLoc='upper left'):
    """
    Plot a pilottone.py results file (.npz) and save it as a PNG.

    npzPath:          results file written by pilottone.py
    pngPath:          output file (default: npzPath with .png instead of .npz)
    tclMotPath:       TCL (TracSuite) motion file, e.g. '/path/2026-04-29_ptoneH20260429_120616_MOT.tsm'.
                      If set, framewise 3D head motion from it is overlaid
    channels:         channel indices to plot (default: all)
    plot:             'both', 'amplitude' or 'phase'
    amplitude:        'relative' (relative_amplitude, default) or 'raw' (amplitude)
    showQuality:      overlay the quality metric
    qualityThreshold: if set, lines with quality below this are left as gaps
    legendLoc:        matplotlib legend location, or 'none' for no legend

    Returns the PNG path.
    """
    if pngPath is None:
        pngPath = os.path.splitext(npzPath)[0] + '.png'
    if plot not in ('both', 'amplitude', 'phase'):
        raise ValueError("plot must be 'both', 'amplitude' or 'phase' (got %r)" % (plot,))
    if amplitude not in ('relative', 'raw'):
        raise ValueError("amplitude must be 'relative' or 'raw' (got %r)" % (amplitude,))

    d = np.load(npzPath)
    settings = json.loads(str(d['settings'])) if 'settings' in d.files else {}
    lineNumber = d['scan_counter']
    amp = d['relative_amplitude'] if amplitude == 'relative' else d['amplitude']
    phase = d['relative_phase']
    quality = d['quality']
    numChanTotal = phase.shape[1]
    # The x-axis spans the whole scan, not just the analysed lines
    # (older files have no last_scan_counter; -1 means unknown)
    lastLine = int(d['last_scan_counter']) if 'last_scan_counter' in d.files else -1
    lastLine = max(lastLine, int(lineNumber.max()))

    chanIndices = list(range(numChanTotal)) if channels is None else list(channels)
    for c in chanIndices:
        if not 0 <= c < numChanTotal:
            raise ValueError("channel %d out of range [0, %d]" % (c, numChanTotal - 1))

    # Low quality lines left as gaps
    ampPlot = amp.astype(float).copy()
    phasePlot = phase.astype(float).copy()
    if qualityThreshold is not None:
        bad = quality < qualityThreshold
        ampPlot[bad, :] = np.nan
        phasePlot[bad, :] = np.nan
    # Y-limits are set from lines at least this good, as for the phase in fast_phase_inspect.py
    yLimQuality = qualityThreshold if qualityThreshold is not None else 0.5

    motion = None
    if tclMotPath is not None:
        tcl = load_tcl_module()
        motion = tcl.match_tcl(tcl.read_tcl(tclMotPath), d['time_ms'])['3D motion framewise'].values

    refChan = settings.get('refChanIdx')
    txStartLine = settings.get('ptoneTxStartScanCounter')

    featureCols = ['amplitude', 'phase'] if plot == 'both' else [plot]
    numCols = len(featureCols)
    figHeightIn = 3 * len(chanIndices) + 1
    fig = Figure(figsize=(8 * numCols, figHeightIn))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(chanIndices), numCols, sharex=True, squeeze=False)

    for row, chan in enumerate(chanIndices):
        chanLabel = 'Channel %d%s' % (chan, ' (reference)' if chan == refChan else '')
        for col, feature in enumerate(featureCols):
            ax = axes[row, col]
            if feature == 'amplitude':
                series, color = ampPlot[:, chan], 'tab:blue'
                label = 'Relative amplitude' if amplitude == 'relative' else 'Amplitude'
                # Constant (e.g. the reference channel's relative amplitude of 1): +/- 10%
                set_ylim_to_data(ax, amp[:, chan], quality >= yLimQuality,
                                 lambda centre: 0.1 * abs(centre) if centre != 0 else 1.0)
            else:
                series, color, label = phasePlot[:, chan], 'tab:green', 'Relative phase'
                # Constant (e.g. the reference channel's relative phase of 0): +/- 1 rad
                set_ylim_to_data(ax, phase[:, chan], quality >= yLimQuality, lambda centre: 1.0)

            ax.plot(lineNumber, series, color=color, label=label)
            ax.set_ylabel(label, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            ax.set_title('%s: %s' % (chanLabel, label))
            if txStartLine is not None:
                ax.axvline(txStartLine, color='gray', linestyle='--', linewidth=1, label='Transmitter start')

            extraLines = []
            if showQuality:
                axQ = ax.twinx()
                axQ.plot(lineNumber, quality, color='tab:orange', label='Quality')
                axQ.set_ylabel('Quality', color='tab:orange')
                axQ.set_ylim(0.0, 1.0)
                axQ.tick_params(axis='y', labelcolor='tab:orange')
                extraLines += axQ.get_lines()
            if motion is not None:
                axM = ax.twinx()
                # Push the motion axis outward so it doesn't overlap the quality axis
                axM.spines['right'].set_position(('axes', 1.12 if showQuality else 1.0))
                axM.plot(lineNumber, motion, color='tab:red', label='3D motion framewise (mm)', alpha=0.7)
                axM.set_ylabel('3D motion framewise (mm)', color='tab:red')
                axM.tick_params(axis='y', labelcolor='tab:red')
                extraLines += axM.get_lines()

            if legendLoc and legendLoc.lower() != 'none':
                allLines = ax.get_lines() + extraLines
                ax.legend(allLines, [l.get_label() for l in allLines], loc=legendLoc)

    for col in range(numCols):
        axes[-1, col].set_xlabel('Line (scan_counter)')
        axes[-1, col].set_xlim(0, lastLine)

    # Title from the results file
    header = str(d['mrd_header']) if 'mrd_header' in d.files else ''
    match = re.search(r'<protocolName>(.*?)</protocolName>', header)
    title = [match.group(1) if match else os.path.basename(npzPath)]
    if 'timestamp' in d.files:
        title.append(str(d['timestamp']))
    if settings.get('ptoneTxFreqHz') is not None:
        title.append('tx %.0f Hz, %g dB%s' % (settings['ptoneTxFreqHz'], settings.get('ptoneTxDB', float('nan')),
                                               '' if settings.get('ptoneTx') else ' (transmitter off)'))
    if settings.get('gitCommit'):
        title.append('git ' + short_git_commit(settings['gitCommit']))
    # Title a fixed distance from the top.  The default position is a fraction of the figure height,
    # which for a tall (many channel) figure puts it several inches down, over the first plots
    titleBandIn = 0.5
    fig.suptitle(' | '.join(title), y=1 - 0.1 / figHeightIn, va='top')

    fig.tight_layout(rect=(0, 0, 1, 1 - titleBandIn / figHeightIn))   # Leave the title band free
    if (motion is not None) and (numCols > 1):
        # tight_layout doesn't allow for the motion axis pushed outward, so make room between columns
        fig.subplots_adjust(wspace=0.45)
    fig.savefig(pngPath, dpi=100)
    return pngPath

def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot pilot tone results (.npz) saved by pilottone.py")
    parser.add_argument('npzfile', help='Results file from pilottone.py')
    parser.add_argument('--outfile', default=None, help='Output PNG (default: npzfile with .png)')
    parser.add_argument('--tcl', default=None, metavar='MOTFILE',
                        help='TCL (TracSuite) motion file (*_MOT.tsm) to overlay framewise 3D head motion, '
                             'e.g. /path/2026-04-29_ptoneH20260429_120616_MOT.tsm')
    parser.add_argument('--channel', type=int, nargs='+', default=None, metavar='N',
                        help='Plot only these channels (default: all)')
    parser.add_argument('--plot', choices=('both', 'amplitude', 'phase'), default='both',
                        help='Which plots to draw per channel (default: both)')
    parser.add_argument('--amplitude', choices=('relative', 'raw'), default='relative',
                        help='relative_amplitude (default) or raw amplitude')
    parser.add_argument('--no-quality', action='store_true', help="Don't overlay the quality metric")
    parser.add_argument('--quality-threshold', type=float, default=None, metavar='X',
                        help='Leave lines with quality below X as gaps')
    parser.add_argument('--legend-loc', default='upper left',
                        help="matplotlib legend location, e.g. 'upper right', or 'none' (default: 'upper left')")
    args = parser.parse_args(argv)

    if args.tcl is not None and not os.path.isfile(args.tcl):
        print("ptone_plot.py: ERROR: TCL motion file not found: %s" % args.tcl, file=sys.stderr)
        return 1
    try:
        pngPath = plot_npz(args.npzfile, pngPath=args.outfile, tclMotPath=args.tcl, channels=args.channel,
                           plot=args.plot, amplitude=args.amplitude, showQuality=not args.no_quality,
                           qualityThreshold=args.quality_threshold, legendLoc=args.legend_loc)
    except (ValueError, OSError) as e:
        print("ptone_plot.py: ERROR: %s" % e, file=sys.stderr)
        return 1
    print("ptone_plot.py: Saved plot to %s" % pngPath)
    return 0

if __name__ == '__main__':
    sys.exit(main())
