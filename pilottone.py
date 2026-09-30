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

    try:
        for item in connection:
            if item is None:
                break

            if not isinstance(item, ismrmrd.Acquisition):
                continue

            # Skip non-imaging lines
            if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) or item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                continue

            results.append(analyze_line(item))

    finally:
        save_results(results)
        connection.send_close()

def analyze_line(acq):
    # acq.data is complex64 with shape [channels, readout samples]
    
    param_est, quality = estimate_ptone_params_initial_np_linalg_svd(np.transpose(acq.data))
    
    result = {
        'scan_counter': acq.scan_counter,
        'line':         acq.idx.kspace_encode_step_1,
        'slice':        acq.idx.slice,
        'time_ms':      acq.acquisition_time_stamp * 2.5,  # Siemens timestamps are in 2.5 ms ticks
        'amplitude':    param_est[0,:],                    # One value per channel
        'phase':        param_est[1,:],                    # One value per channel (radians)
        'quality':      quality,
        # Relative to channel 0, removing the per-line scale/phase ambiguity of the SVD estimate
        'relative_amplitude': param_est[0,:] / param_est[0,0],
        'relative_phase':     np.angle(np.exp(1j * (param_est[1,:] - param_est[1,0]))),  # Wrapped to [-pi, pi]
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
