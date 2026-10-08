#!/usr/bin/env python3
# Plot k-space lines with the pilot tone fitted (ptone/tone_estimation.py) and subtracted (ptone/tone_removal.py),
# as a PNG.
#
# For each chosen line and channel, one row of panels:
#   - k-space: the line's samples (real part, imaginary part or magnitude), the fitted tone, and the line
#     after subtracting the tone;
#   - spectrum: the FFT magnitude (dB) of the same three, against frequency (kHz from the centre of the
#     readout band, with FFT bins on the top axis), with the fitted tone frequency marked (and the nominal
#     one, from the transmit frequency, if --tx-freq is given), and the imaging band shaded;
#   - optionally (--zoom-bins), the spectrum zoomed in around the tone.
# The tone is fitted from all channels of the line together, whichever channels are plotted.
#
# Reads Siemens raw data (twix .dat, with twixtools; the last measurement, normally the imaging scan, unless
# --meas says otherwise) or MRD (.mrd/.h5, e.g. from siemens_to_ismrmrd).  Lines are chosen by scan counter
# (scan_counter in MRD; ScanCounter in twix), the numbering ptone_plot.py uses on its x-axis.
#
# Command line:
#   ptone-tone-plot <meas.dat | file.mrd> --lines 120 450 900 --channels 0 12 [--out plot.png]
#                   [--meas -1] [--part real|imag|abs] [--zoom-bins 20] [--tx-freq 123.285661e6] [--dpi 200]
#   (or python ptone/tone_plot.py ...)
# Python:
#   from ptone.tone_plot import plot_lines
#   plot_lines('meas.dat', lines=[120, 450], channels=[0, 12])

import argparse
import os
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    from ptone.tone_estimation import fit_tone, tone_model
    from ptone.tone_removal import subtract_tone, band_power_reduction_db
except ImportError:                 # Run as a script from ptone/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ptone.tone_estimation import fit_tone, tone_model
    from ptone.tone_removal import subtract_tone, band_power_reduction_db

# ----- Reading lines ----------------------------------------------------------------------------

def is_twix(path):
    return path.lower().endswith('.dat')

def read_mrd_lines(path, scanCounters):
    """
    The lines with the given scan counters from an MRD file, and what's needed about the scan.  Returns
    (info, lines): info has 'protocol', 'f0Hz', 'imagingHalfBand' (cycles per sample) and 'header'; each line
    is a dict with 'scanCounter', 'data' (channels x samples), 'dwellS', and 'nominal' (a function giving
    the tone's nominal frequency in Hz from the band centre for a transmit frequency).
    """
    import h5py
    import ismrmrd
    try:
        from ptone.tx_frequency import ptone_readout_offset
    except ImportError:
        from tx_frequency import ptone_readout_offset
    with h5py.File(path, 'r') as f:
        groups = list(f.keys())
        group = 'dataset' if 'dataset' in groups else groups[0]
        xml = f[group]['xml'][0]
    header = ismrmrd.xsd.CreateFromDocument(xml)
    enc = header.encoding[0]
    info = dict(header=header, protocol=getattr(header.measurementInformation, 'protocolName', None) or '',
                f0Hz=header.experimentalConditions.H1resonanceFrequency_Hz,
                imagingHalfBand=0.5 * enc.reconSpace.fieldOfView_mm.x / enc.encodedSpace.fieldOfView_mm.x)
    wanted, lines = set(scanCounters), {}
    ds = ismrmrd.Dataset(path, group, create_if_needed=False)
    try:
        for i in range(ds.number_of_acquisitions()):
            acq = ds.read_acquisition(i)
            if acq.scan_counter in wanted and acq.scan_counter not in lines:
                lines[acq.scan_counter] = dict(
                    scanCounter=int(acq.scan_counter), data=np.array(acq.data), dwellS=acq.sample_time_us * 1e-6,
                    nominal=(lambda txHz, acq=acq: ptone_readout_offset(txHz, header, acq)))
                if len(lines) == len(wanted):
                    break
    finally:
        ds.close()
    return info, lines

def read_twix_lines(path, scanCounters, meas=-1):
    """As read_mrd_lines(), from a Siemens twix .dat file (measurement meas: default the last)."""
    try:
        import twixtools
    except ImportError:
        raise ImportError("Reading .dat files needs twixtools (pip install twixtools); or convert to MRD with siemens_to_ismrmrd")
    twix = twixtools.read_twix(path, parse_pmu=False, keep_syncdata=False, verbose=False)
    m = twix[meas]
    yaps = m['hdr']['MeasYaps']
    dwellS = yaps['sRXSPEC']['alDwellTime'][0] * 1e-9
    baseResolution = yaps['sKSpace']['lBaseResolution']
    readoutFovMm = yaps['sSliceArray']['asSlice'][0]['dReadoutFOV']
    f0Hz = yaps['sTXSPEC']['asNucleusInfo'][0]['lFrequency']
    info = dict(header=m['hdr'], protocol=str(m['hdr']['Meas'].get('tProtocolName', '') if 'Meas' in m['hdr'] else ''),
                f0Hz=f0Hz, imagingHalfBand=None)
    wanted, lines = set(scanCounters), {}
    for mdb in m['mdb']:
        sc = int(mdb.mdh.ScanCounter)
        if sc in wanted and sc not in lines:
            data = np.array(mdb.data)
            numSamples = data.shape[1]
            # Encoded readout FOV includes the oversampling (samples / base resolution, normally 2)
            encodedFovMm = readoutFovMm * numSamples / baseResolution
            offsetMm = float(mdb.mdh.ReadOutOffcentre)
            hzPerMm = 1 / (dwellS * encodedFovMm)
            lines[sc] = dict(scanCounter=sc, data=data, dwellS=dwellS,
                             nominal=(lambda txHz, offsetMm=offsetMm, hzPerMm=hzPerMm: txHz - f0Hz - offsetMm * hzPerMm))
            if info['imagingHalfBand'] is None:
                info['imagingHalfBand'] = 0.5 * baseResolution / numSamples
            if len(lines) == len(wanted):
                break
    if info['imagingHalfBand'] is None:
        info['imagingHalfBand'] = 0.25
    return info, lines

def read_lines(path, scanCounters, meas=-1):
    """read_twix_lines() or read_mrd_lines(), by file type.  Raises ValueError for missing lines."""
    info, lines = read_twix_lines(path, scanCounters, meas) if is_twix(path) else read_mrd_lines(path, scanCounters)
    missing = [sc for sc in scanCounters if sc not in lines]
    if missing:
        raise ValueError("Line(s) %s not found in %s" % (', '.join(map(str, missing)), path))
    return info, lines

# ----- Plotting ---------------------------------------------------------------------------------

def wrap_cycles(x):
    return (x + 0.5) % 1.0 - 0.5

def plot_lines(path, lines, channels=(0,), pngPath=None, meas=-1, part='real', zoomBins=0, txFreqHz=None, dpi=200):
    """
    Plot the given lines (scan counters) and channels of a twix .dat or MRD file, with the pilot tone fitted
    and subtracted, and save a PNG (at dpi dots per inch).  Returns (pngPath, fits): fits[scanCounter] is fit_tone()'s result,
    plus 'freqHz' (the tone's frequency in Hz from the band centre).
    """
    if part not in ('real', 'imag', 'abs'):
        raise ValueError("part must be 'real', 'imag' or 'abs' (got %r)" % (part,))
    lines, channels = [int(l) for l in lines], [int(c) for c in channels]
    if pngPath is None:
        stem = os.path.splitext(path)[0]
        pngPath = '%s--tone--lines-%s.png' % (stem, '-'.join(map(str, lines[:6])) + ('-etc' if len(lines) > 6 else ''))
    info, data = read_lines(path, lines, meas)
    for sc in lines:
        numChan = data[sc]['data'].shape[0]
        bad = [c for c in channels if not 0 <= c < numChan]
        if bad:
            raise ValueError("Channel(s) %s out of range for line %d (%d channels)" % (bad, sc, numChan))

    fits = {}
    numCols = 3 if zoomBins else 2
    rows = [(sc, c) for sc in lines for c in channels]
    figHeightIn = 3.2 * len(rows) + 1
    fig = Figure(figsize=(7 * numCols, figHeightIn))
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(rows), numCols, squeeze=False)
    partOf = {'real': np.real, 'imag': np.imag, 'abs': np.abs}[part]
    colours = {'data': 'tab:blue', 'fitted tone': 'tab:orange', 'filtered': 'tab:green'}
    # The fitted tone is dashed and drawn on top, so it's easy to see how it lines up with the data
    styles = {'data': dict(linewidth=0.8, alpha=0.7, zorder=2),
              'fitted tone': dict(linewidth=1.0, linestyle='--', zorder=4),
              'filtered': dict(linewidth=0.8, zorder=3)}

    for row, (sc, chan) in enumerate(rows):
        line = data[sc]
        y = line['data']
        numSamples = y.shape[1]
        if sc not in fits:
            fits[sc] = fit_tone(y, imagingHalfBand=info['imagingHalfBand'])
            fits[sc]['freqHz'] = fits[sc]['freq'] / line['dwellS']
        fit = fits[sc]
        tone = tone_model(fit['freq'], fit['amplitudes'], numSamples)
        filtered = subtract_tone(y, fit)
        traces = [('data', y[chan]), ('fitted tone', tone[chan]), ('filtered', filtered[chan])]
        dwellS = line['dwellS']
        toneHz = fit['freq'] / dwellS
        amp = fit['amplitudes'][chan]
        reduction = band_power_reduction_db(y[chan], filtered[chan], fit['freq'])
        nominalHz = None
        if txFreqHz is not None:
            # Wrapped into the readout band, as the sampled data sees it
            nominalHz = wrap_cycles(line['nominal'](txFreqHz) * dwellS) / dwellS

        # k-space
        ax = axes[row, 0]
        n = np.arange(numSamples)
        for (label, trace), colour in zip(traces, colours.values()):
            ax.plot(n, partOf(trace), color=colour, label=label, **styles[label])
        ax.set_xlabel('Sample')
        ax.set_ylabel('%s(k-space)' % {'real': 'Re', 'imag': 'Im', 'abs': '|.|'}[part])
        ax.set_xlim(0, numSamples - 1)
        ax.legend(loc='upper right', fontsize=8)

        # Spectrum (centred: frequency from the band centre)
        freqsKHz = (np.arange(numSamples) - numSamples // 2) / (numSamples * dwellS) / 1000
        spectra = [(label, 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(trace))) + 1e-30)) for label, trace in traces]
        for col in range(1, numCols):
            ax = axes[row, col]
            for (label, spec), colour in zip(spectra, colours.values()):
                ax.plot(freqsKHz, spec, color=colour, label=label, **styles[label])
            halfImagingKHz = info['imagingHalfBand'] / dwellS / 1000
            ax.axvspan(-halfImagingKHz, halfImagingKHz, color='0.92', zorder=0, label='imaging band')
            ax.axvline(toneHz / 1000, color='tab:red', linestyle='--', linewidth=0.8, label='fitted tone %+.1f Hz' % toneHz)
            if nominalHz is not None:
                ax.axvline(nominalHz / 1000, color='0.4', linestyle=':', linewidth=1.0, label='nominal %+.1f Hz' % nominalHz)
            ax.set_xlabel('Frequency from band centre (kHz); top axis: FFT bin')
            ax.set_ylabel('|FFT| (dB)')
            floor = np.percentile(spectra[0][1], 1)
            top = max(s.max() for _, s in spectra)
            ax.set_ylim(floor - 10, top + 5)
            if col == 1:
                ax.set_xlim(freqsKHz[0], freqsKHz[-1])
            else:
                binKHz = 1 / (numSamples * dwellS) / 1000
                ax.set_xlim(toneHz / 1000 - zoomBins * binKHz, toneHz / 1000 + zoomBins * binKHz)
                ax.set_title('Zoom: +/- %d bins around the tone' % zoomBins, fontsize=9, pad=24)
            binsAxis = ax.secondary_xaxis('top', functions=(lambda k, d=dwellS, N=numSamples: k * 1000 * N * d,
                                                            lambda b, d=dwellS, N=numSamples: b / (1000 * N * d)))
            binsAxis.tick_params(labelsize=7)
            if col == 1:
                ax.legend(loc='lower left', fontsize=7)

        axes[row, 0].set_title(
            'Line %d, channel %d: tone %+.1f Hz (%.6f cycles/sample, bin %+.3f), %.1f dB above the noise (all channels)\n'
            '|a| %.3g, phase %.1f deg; power within 3 bins of the tone down %.1f dB'
            % (sc, chan, toneHz, fit['freq'], fit['freq'] * numSamples, fit['peakToNoiseDb'], abs(amp),
               np.degrees(np.angle(amp)), reduction),
            loc='left', fontsize=9)

    title = [os.path.basename(path)]
    if info['protocol']:
        title.append(info['protocol'])
    first = data[lines[0]]
    title.append('%d channels x %d samples, dwell %.2f us' % (first['data'].shape[0], first['data'].shape[1], first['dwellS'] * 1e6))
    if txFreqHz is not None:
        title.append('tx %.0f Hz, f0 %.0f Hz' % (txFreqHz, info['f0Hz']))
    fig.suptitle(' | '.join(title), y=1 - 0.1 / figHeightIn, va='top')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / figHeightIn))
    fig.savefig(pngPath, dpi=dpi)
    return pngPath, fits

def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot k-space lines with the pilot tone fitted and subtracted")
    parser.add_argument('path', help='Siemens raw data (.dat) or MRD file (.mrd/.h5)')
    parser.add_argument('--lines', type=int, nargs='+', required=True, help='Scan counters of the lines to plot')
    parser.add_argument('--channels', type=int, nargs='+', default=[0], help='Channels to plot (default: 0)')
    parser.add_argument('--out', default=None, help='PNG to save (default: next to the input)')
    parser.add_argument('--meas', type=int, default=-1, help='Measurement in a .dat file (default: -1, the last)')
    parser.add_argument('--part', choices=['real', 'imag', 'abs'], default='real',
                        help='Part of the k-space signal to plot (default: real)')
    parser.add_argument('--zoom-bins', type=int, default=0, help='Add a spectrum panel zoomed to +/- this many bins around the tone')
    parser.add_argument('--tx-freq', type=float, default=None, help='Transmit frequency (Hz), to mark the nominal tone frequency')
    parser.add_argument('--dpi', type=int, default=200, help='PNG resolution, dots per inch (default: 200)')
    args = parser.parse_args(argv)
    try:
        pngPath, fits = plot_lines(args.path, args.lines, args.channels, args.out, args.meas, args.part,
                                   args.zoom_bins, args.tx_freq, args.dpi)
    except (ValueError, ImportError) as e:
        print("ptone-tone-plot: %s" % e, file=sys.stderr)
        return 1
    for sc, fit in fits.items():
        print("line %d: tone at %.6f cycles/sample, %+.1f Hz from the band centre, %.1f dB above the noise"
              % (sc, fit['freq'], fit['freqHz'], fit['peakToNoiseDb']))
    print("Saved %s" % pngPath)
    return 0

if __name__ == '__main__':
    sys.exit(main())
