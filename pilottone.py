import ismrmrd
import os
import logging
import numpy as np
import mrdhelper

import sys
import traceback
import time

# Folder for debug output files
debugFolder = "/tmp/share/debug"

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
    param_est = np.zeros((3,num_chan))
    u, s, vh = np.linalg.svd(scan_data)
    quality = s[0]/np.sum(s)
    #ptone_signal = u[:,0] / u[0,0]
    vh_prime = vh[0,:] * s[0] * u[0,0]
    # Amplitude
    param_est[0,:] = np.abs(vh_prime)    
    # Phase
    param_est[1,:] = np.angle(vh_prime)
    # todo: compute error estimate as the first singular value divided by the sum of the singular values
    return param_est, quality

def process(connection, config, mrdHeader):
    logging.info("Config: \n%s", config)

    results = []  # Per-line analysis results

    # The pilot tone transmitter is enabled when the first k-space line is received, so lines
    # within ptoneTxDelayMs of the first line are skipped while it starts up
    ptoneTxDelayMs = mrdhelper.get_json_config_param(config, 'ptoneTxDelayMs', default=0, type='float')
    firstTimeMs  = None
    logging.info("Skipping lines within %g ms of the first line", ptoneTxDelayMs)

    # Per-channel midpoint of the phase range, set from the first line after ptoneTxDelayMs.
    # Keeping phases within [midpoint - pi, midpoint + pi] minimizes phase wraps across the scan
    phaseMidpoint = None

    try:
        for item in connection:
            if item is None:
                break

            if not isinstance(item, ismrmrd.Acquisition):
                continue

            # Skip non-imaging lines
            if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) or item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                continue

            # Skip lines until the pilot tone is on.  Timestamps are 2.5 ms ticks since midnight,
            # so take the difference modulo one day in case the scan crosses midnight
            timeMs = item.acquisition_time_stamp * 2.5
            if firstTimeMs is None:
                firstTimeMs = timeMs
            if (timeMs - firstTimeMs) % (24*60*60*1000) < ptoneTxDelayMs:
                continue

            result = analyze_line(item, phaseMidpoint)
            results.append(result)

            if phaseMidpoint is None:
                phaseMidpoint = result['relative_phase']
                logging.info("Setting phase range midpoint to %s", phaseMidpoint)

    finally:
        save_results(results)
        connection.send_close()

def analyze_line(acq, phaseMidpoint=None):
    # acq.data is complex64 with shape [channels, readout samples]

    param_est, quality = estimate_ptone_params_initial_np_linalg_svd(np.transpose(acq.data))

    amplitude = param_est[0,:]
    phase = np.mod(param_est[1,:], 2*np.pi)
    
    # Phase relative to channel 0, wrapped to [midpoint - pi, midpoint + pi].
    # Without a midpoint (first line), wrap to [0, 2pi] as in fast_phase_avg.py
    phaseDiff = phase - phase[0]
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
        # Relative to channel 0, removing the per-line scale/phase ambiguity of the SVD estimate
        'relative_amplitude': amplitude / amplitude[0],
        'relative_phase':     relativePhase,
    }

    logging.debug("Line %4d (scan %5d): mean amplitude %g, quality %.3f", result['line'], result['scan_counter'], np.mean(result['amplitude']), quality)
    return result

def save_results(results):
    if len(results) == 0:
        return

    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    filePath = os.path.join(debugFolder, "pilottone.npz")
    np.savez(filePath,
             scan_counter = np.array([r['scan_counter'] for r in results]),
             line         = np.array([r['line']         for r in results]),
             slice        = np.array([r['slice']        for r in results]),
             time_ms      = np.array([r['time_ms']      for r in results]),
             amplitude    = np.stack([r['amplitude']    for r in results]),  # [lines, channels]
             phase        = np.stack([r['phase']        for r in results]),  # [lines, channels]
             quality      = np.array([r['quality']      for r in results]),
             relative_amplitude = np.stack([r['relative_amplitude'] for r in results]),  # [lines, channels]
             relative_phase     = np.stack([r['relative_phase']     for r in results]))  # [lines, channels]
    logging.info("Saved pilot tone results for %d lines to %s", len(results), filePath)
