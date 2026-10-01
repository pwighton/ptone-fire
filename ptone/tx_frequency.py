# Pilot tone transmit frequency, calculated from the MRD header and an acquisition header.
#
# The readout (with oversampling) covers f0 +/- 1/(2*dwell).  The imaging band (the reconstructed
# FOV) is the central part of that, f0 +/- reconFOV/encodedFOV * 1/(2*dwell).  The pilot tone is
# placed bandPosition of the way from the edge of the imaging band to the edge of the readout band,
# on the high-frequency side.  With 2x readout oversampling and bandPosition = 0.5 (halfway) this is
# f0 + 0.75 * baseResolution * bandwidthPerPixel.
#
# The scanner shifts its receive frequency to move the FOV along the readout direction, so a tone
# at a fixed frequency appears at (txFrequency - f0 - fovShift) in the readout.  The fovShift is
# added to the transmit frequency to keep the tone at the same place relative to the FOV.
# Verified on ptone20260114b (readout offset +22.27 mm) and ptoneH20260429 (-8.37 mm) to within
# half a frequency bin (see tests/test_tx_frequency.py).

import numpy as np

def readout_frequencies(mrdHeader, acq):
    # Frequencies (Hz, relative to f0) describing the readout of acq.  Uses the first encoding
    # in the header, and the dwell time and position from the acquisition header
    encoding    = mrdHeader.encoding[0]
    dwellS      = acq.sample_time_us * 1e-6
    encodedFovX = encoding.encodedSpace.fieldOfView_mm.x
    reconFovX   = encoding.reconSpace.fieldOfView_mm.x

    readoutHalfBandHz = 1 / (2 * dwellS)
    imagingHalfBandHz = readoutHalfBandHz * reconFovX / encodedFovX
    hzPerMm           = (1 / dwellS) / encodedFovX
    readoutOffsetMm   = float(np.dot(acq.position, acq.read_dir))

    return {
        'f0Hz':              mrdHeader.experimentalConditions.H1resonanceFrequency_Hz,
        'dwellS':            dwellS,
        'readoutHalfBandHz': readoutHalfBandHz,
        'imagingHalfBandHz': imagingHalfBandHz,
        'hzPerMm':           hzPerMm,
        'readoutOffsetMm':   readoutOffsetMm,
        'fovShiftHz':        readoutOffsetMm * hzPerMm,
    }

def ptone_tx_frequency(mrdHeader, acq, bandPosition=0.5):
    # Transmit frequency (Hz) that places the pilot tone bandPosition of the way from the edge of
    # the imaging band (0) to the edge of the readout band (1)
    r = readout_frequencies(mrdHeader, acq)
    toneOffsetHz = r['imagingHalfBandHz'] + bandPosition * (r['readoutHalfBandHz'] - r['imagingHalfBandHz'])
    return r['f0Hz'] + toneOffsetHz + r['fovShiftHz']

def ptone_readout_offset(txFrequencyHz, mrdHeader, acq):
    # Where a tone transmitted at txFrequencyHz appears in the readout of acq, in Hz from the
    # centre of the readout band (the inverse of ptone_tx_frequency)
    r = readout_frequencies(mrdHeader, acq)
    return txFrequencyHz - r['f0Hz'] - r['fovShiftHz']
