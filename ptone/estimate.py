# Pilot tone parameter estimation, used by pilottone.py.
# img_band_stop and estimate_ptone_params_initial_np_linalg_svd come from kstream's fast_phase_avg.py

import logging
import numpy as np

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
