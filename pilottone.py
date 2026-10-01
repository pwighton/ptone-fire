import ismrmrd
import os
import json
import re
import logging
import threading
import numpy as np
import mrdhelper

import sys
import traceback
import time
from datetime import datetime

# Defaults for the outputFolder and outputFileStem parameters in pilottone.json.  Results are saved to
# <outputFolder>/<outputFileStem>--<protocolName>--<YYYYMMDD-HHMMSS-mmm>.npz, timestamped when processing starts
defaultOutputFolder   = "/tmp/ismrmrd-server-output--pilottone"
defaultOutputFileStem = "pilottone"

fft  = lambda x, ax : np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=ax), norm='ortho', axis=ax), axes=ax)
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X, axes=ax), norm='ortho', axis=ax), axes=ax)

def img_band_stop(scan_data):
    num_samp = scan_data.shape[0]
    lo = num_samp // 4
    hi = 3 * num_samp // 4
    filt = np.ones(num_samp, dtype=scan_data.dtype)
    filt[lo:hi] = 0.0
    filt = filt[:, np.newaxis]
    return ifft(fft(scan_data, 0) * filt, 0)

def estimate_ptone_params_initial_np_linalg_svd(scan_data, filter_img_data=True):
    if filter_img_data:
        scan_data = img_band_stop(scan_data)
    num_samp, num_chan = scan_data.shape
    # parameters per channel (amplitude, phase, frequency)
    # We are not estimating frequency to save time
    param_est = np.zeros((3,num_chan))
    u, s, vh = np.linalg.svd(scan_data)
    quality = s[0]/np.sum(s)
    #ptone_signal = u[:,0] / u[0,0]
    vh_prime = vh[0,:] * s[0] * u[0,0]
    # Amplitude
    param_est[0,:] = np.abs(vh_prime)    
    # Phase
    param_est[1,:] = np.angle(vh_prime)
    return param_est, quality

# Defaults for the skipFlags and dontSkipFlags parameters in pilottone.json.  These are the equivalent
# of is_image_scan() in kstream's twixtools_mdh.py, using ISMRMRD flags.
# Siemens flags with no ISMRMRD equivalent (SLICE_ACCEL_REFSCAN, SLICE_ACCEL_PHASCOR, noname60)
# can't be checked.  SYNCDATA arrives as waveforms and ACQEND has no acquisition, so neither is needed
defaultSkipFlags = [
    'ACQ_IS_RTFEEDBACK_DATA',                 # RTFEEDBACK
    'ACQ_IS_HPFEEDBACK_DATA',                 # HPFEEDBACK
    'ACQ_IS_PHASE_STABILIZATION_REFERENCE',   # REFPHASESTABSCAN
    'ACQ_IS_PHASE_STABILIZATION',             # PHASESTABSCAN
    'ACQ_IS_PHASECORR_DATA',                  # PHASCOR
    'ACQ_IS_NOISE_MEASUREMENT',               # NOISEADJSCAN
    'ACQ_IS_PARALLEL_CALIBRATION',            # PATREFSCAN
    # Not in twixtools_mdh.py.  fast_phase_avg.py skips the whole AdjCoilSens measurement instead
    'ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA',
    'ACQ_IS_NAVIGATION_DATA',
]

# Defaults for the dontSkipFlags
# As in twixtools_mdh.py, PATREFANDIMASCAN lines are image lines regardless of other flags
defaultDontSkipFlags = [
    'ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING',  # PATREFANDIMASCAN
]

def get_flags_config_param(config, key, default):
    # Read a list of ISMRMRD flag names from the JSON config, as either a JSON list or a
    # comma-separated string, and convert them to flag values.  Unknown names raise an error
    names = default
    if isinstance(config, dict) and key in config.get('parameters', {}):
        names = config['parameters'][key]
        if isinstance(names, str):
            names = [name.strip() for name in names.split(',') if name.strip() != '']

    unknown = [name for name in names if not (name.startswith('ACQ_') and isinstance(getattr(ismrmrd, name, None), int))]
    if len(unknown) > 0:
        raise ValueError("Unknown ISMRMRD flag(s) in '%s': %s" % (key, ', '.join(unknown)))

    return [getattr(ismrmrd, name) for name in names], names

def is_image_line(acq, skipFlags, dontSkipFlags):
    # Lines with any dontSkipFlags set are image lines; otherwise lines with any skipFlags set are not
    if any(acq.is_flag_set(flag) for flag in dontSkipFlags):
        return True
    return not any(acq.is_flag_set(flag) for flag in skipFlags)

def get_protocol_name(mrdHeader):
    # Protocol name from the MRD header, with characters other than letters, digits, '.', '_' and '-'
    # replaced by '_' so it's safe in a filename.  The server passes the header as text if it isn't valid MRD XML
    try:
        protocolName = mrdHeader.measurementInformation.protocolName
    except AttributeError:
        protocolName = None
    if not protocolName:
        return 'unknown'
    return re.sub(r'[^A-Za-z0-9._-]', '_', protocolName)

def mrd_header_to_xml(mrdHeader):
    # MRD header as XML text for saving.  The server passes the header as text if it isn't valid MRD XML
    if isinstance(mrdHeader, ismrmrd.xsd.ismrmrdHeader):
        return ismrmrd.xsd.ToXML(mrdHeader)
    if mrdHeader is None:
        return ''
    return str(mrdHeader)

def process(connection, config, mrdHeader):
    results = []  # Per-line analysis results

    # Output file for the results
    outputFolder   = mrdhelper.get_json_config_param(config, 'outputFolder',   default=defaultOutputFolder,   type='str')
    outputFileStem = mrdhelper.get_json_config_param(config, 'outputFileStem', default=defaultOutputFileStem, type='str')
    now            = datetime.now()
    timestamp      = now.strftime('%Y%m%d-%H%M%S') + '-%03d' % (now.microsecond // 1000)  # YYYYMMDD-HHMMSS-mmm
    protocolName   = get_protocol_name(mrdHeader)
    outputFilePath = os.path.join(outputFolder, outputFileStem + '--' + protocolName + '--' + timestamp + '.npz')

    # Also write this scan's log messages to a .txt file alongside the results.  Only messages from
    # this thread are included, in case the server is handling other connections at the same time
    logFilePath = os.path.splitext(outputFilePath)[0] + '.txt'
    os.makedirs(outputFolder, exist_ok=True)
    logHandler = logging.FileHandler(logFilePath)
    logHandler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    thisThread = threading.get_ident()
    logHandler.addFilter(lambda record: record.thread == thisThread)
    logging.getLogger().addHandler(logHandler)

    logging.info("Config: \n%s", config)
    logging.info("mrdHeader: \n%s", mrdHeader)
    logging.info("Results will be saved to %s", outputFilePath)
    logging.info("Log will be saved to %s", logFilePath)

    # The pilot tone transmitter is enabled when the first k-space line is received, so lines
    # within ptoneTxDelayMs of the first line are skipped while it starts up
    ptoneTxDelayMs = mrdhelper.get_json_config_param(config, 'ptoneTxDelayMs', default=0, type='float')
    firstTimeMs  = None
    logging.info("Skipping lines within %g ms of the first line", ptoneTxDelayMs)

    # Reference channel for relative phase and amplitude (phase_ref_chan_idx in fast_phase_avg.py)
    refChanIdx = mrdhelper.get_json_config_param(config, 'refChanIdx', default=0, type='int')
    logging.info("Using channel %d as the reference channel", refChanIdx)

    # Per-channel midpoint of the phase range, set from the first line after ptoneTxDelayMs.
    # Keeping phases within [midpoint - pi, midpoint + pi] minimizes phase wraps across the scan
    phaseMidpoint = None

    numChanMismatch = 0  # Lines skipped because their channel count differs from the first analyzed line

    # Settings actually used (config values with defaults filled in), saved with the results
    settings = {
        'outputFolder':   outputFolder,
        'outputFileStem': outputFileStem,
        'ptoneTxDelayMs': ptoneTxDelayMs,
        'refChanIdx':     refChanIdx,
    }

    try:
        # Lines to analyze, based on their ISMRMRD flags (see defaultSkipFlags and defaultDontSkipFlags)
        skipFlags, skipFlagNames = get_flags_config_param(config, 'skipFlags', defaultSkipFlags)
        dontSkipFlags, dontSkipFlagNames = get_flags_config_param(config, 'dontSkipFlags', defaultDontSkipFlags)
        logging.info("Skipping lines with flags: %s", ', '.join(skipFlagNames))
        logging.info("Unless they have flags:    %s", ', '.join(dontSkipFlagNames))
        settings['skipFlags']     = skipFlagNames
        settings['dontSkipFlags'] = dontSkipFlagNames

        for item in connection:
            if item is None:
                break

            if not isinstance(item, ismrmrd.Acquisition):
                continue

            # Skip non-imaging lines
            if not is_image_line(item, skipFlags, dontSkipFlags):
                continue

            # Skip lines until the pilot tone is on.  Timestamps are 2.5 ms ticks since midnight,
            # so take the difference modulo one day in case the scan crosses midnight
            timeMs = item.acquisition_time_stamp * 2.5
            if firstTimeMs is None:
                firstTimeMs = timeMs
            if (timeMs - firstTimeMs) % (24*60*60*1000) < ptoneTxDelayMs:
                continue

            # Skip lines whose channel count differs from the line the phase midpoint was set from
            if (phaseMidpoint is not None) and (item.data.shape[0] != len(phaseMidpoint)):
                if numChanMismatch == 0:
                    logging.warning("Skipping line (scan %d) with %d channels; expected %d.  Further mismatches are counted but not logged",
                                    item.scan_counter, item.data.shape[0], len(phaseMidpoint))
                numChanMismatch += 1
                continue

            result = analyze_line(item, refChanIdx, phaseMidpoint)
            results.append(result)

            if phaseMidpoint is None:
                phaseMidpoint = result['relative_phase']
                logging.info("Setting phase range midpoint to %s", phaseMidpoint)

    except Exception:
        # Logged here rather than left to the server, so the traceback is also in the log file
        logging.exception("pilottone processing failed")

    finally:
        if numChanMismatch > 0:
            logging.warning("Skipped %d lines with a mismatched channel count", numChanMismatch)
        save_results(results, outputFilePath, timestamp, config, settings, mrdHeader)
        connection.send_close()
        logging.getLogger().removeHandler(logHandler)
        logHandler.close()

def analyze_line(acq, refChanIdx=0, phaseMidpoint=None):
    # acq.data is complex64 with shape [channels, readout samples]

    param_est, quality = estimate_ptone_params_initial_np_linalg_svd(np.transpose(acq.data))

    amplitude = param_est[0,:]
    phase = np.mod(param_est[1,:], 2*np.pi)
    
    # Phase relative to the reference channel, wrapped to [midpoint - pi, midpoint + pi].
    # Without a midpoint (first line), wrap to [0, 2pi] as in fast_phase_avg.py
    phaseDiff = phase - phase[refChanIdx]
    if phaseMidpoint is None:
        relativePhase = np.mod(phaseDiff, 2*np.pi)
    else:
        relativePhase = phaseMidpoint + np.angle(np.exp(1j * (phaseDiff - phaseMidpoint)))

    result = {
        'scan_counter': acq.scan_counter,
        'line':         acq.idx.kspace_encode_step_1,
        'slice':        acq.idx.slice,
        'time_ms':      acq.acquisition_time_stamp * 2.5,  # Siemens timestamps are in 2.5 ms ticks
        'amplitude':    amplitude,                    # One value per channel
        'phase':        phase,                    # One value per channel (radians)
        'quality':      quality,
        # Relative to the reference channel, removing the per-line scale/phase ambiguity of the SVD estimate
        'relative_amplitude': amplitude / amplitude[refChanIdx],
        'relative_phase':     relativePhase,
    }

    logging.debug("Line %4d (scan %5d): mean amplitude %g, quality %.3f", result['line'], result['scan_counter'], np.mean(result['amplitude']), quality)
    return result

def save_results(results, filePath, timestamp, config, settings, mrdHeader):
    if len(results) == 0:
        return

    outputFolder = os.path.dirname(filePath)
    if (outputFolder != "") and (not os.path.exists(outputFolder)):
        os.makedirs(outputFolder)
        logging.debug("Created folder " + outputFolder + " for output files")

    np.savez(filePath,
             scan_counter = np.array([r['scan_counter'] for r in results]),
             line         = np.array([r['line']         for r in results]),
             slice        = np.array([r['slice']        for r in results]),
             time_ms      = np.array([r['time_ms']      for r in results]),
             amplitude    = np.stack([r['amplitude']    for r in results]),  # [lines, channels]
             phase        = np.stack([r['phase']        for r in results]),  # [lines, channels]
             quality      = np.array([r['quality']      for r in results]),
             relative_amplitude = np.stack([r['relative_amplitude'] for r in results]),  # [lines, channels]
             relative_phase     = np.stack([r['relative_phase']     for r in results]),  # [lines, channels]
             timestamp    = np.array(timestamp),                                # Processing start, YYYYMMDD-HHMMSS-mmm
             config       = np.array(json.dumps(config, indent=4)),            # Config as received, as JSON text
             settings     = np.array(json.dumps(settings, indent=4)),          # Settings actually used, as JSON text
             mrd_header   = np.array(mrd_header_to_xml(mrdHeader)))            # MRD header, as XML text
    logging.info("Saved pilot tone results for %d lines to %s", len(results), filePath)
