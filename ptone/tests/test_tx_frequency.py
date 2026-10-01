# Tests for ptone/tx_frequency.py
#
# Run from the repository root with:
#   python -m pytest ptone/tests
#
# The dataset tests need the converted .h5 files.  They're looked for in $PTONE_TEST_DATA, or
# ../pilot-tone-test-data relative to the repository root, and skipped if not found.

import os
from types import SimpleNamespace

import h5py
import ismrmrd
import numpy as np
import pytest

from ptone.tx_frequency import readout_frequencies, ptone_tx_frequency, ptone_readout_offset

# ----- Synthetic header and acquisition -------------------------------------------------------

def make_header(f0Hz=123_248_104, encodedFovX=480.0, reconFovX=240.0):
    # Just the header fields tx_frequency.py uses
    return SimpleNamespace(
        experimentalConditions=SimpleNamespace(H1resonanceFrequency_Hz=f0Hz),
        encoding=[SimpleNamespace(
            encodedSpace=SimpleNamespace(fieldOfView_mm=SimpleNamespace(x=encodedFovX)),
            reconSpace=SimpleNamespace(fieldOfView_mm=SimpleNamespace(x=reconFovX)),
        )],
    )

def make_acq(sampleTimeUs=9.8, readoutOffsetMm=0.0, readDir=(0.0, 1.0, 0.0)):
    acq = ismrmrd.Acquisition.from_array(np.zeros((1, 512), dtype=np.complex64))
    acq.sample_time_us = sampleTimeUs
    acq.read_dir = readDir
    acq.position = tuple(readoutOffsetMm * np.asarray(readDir))
    return acq

def test_matches_kstream_formula_without_fov_shift():
    # With 2x oversampling and no FOV shift: f0 + 0.75 * baseResolution * bandwidthPerPixel
    hdr, acq = make_header(), make_acq()
    baseRes = 256
    bwPerPixel = 1 / (512 * 9.8e-6)
    assert ptone_tx_frequency(hdr, acq) == pytest.approx(123_248_104 + 0.75 * baseRes * bwPerPixel)

def test_band_position_endpoints():
    # bandPosition 0 is the edge of the imaging band, 1 is the edge of the readout band
    hdr, acq = make_header(), make_acq()
    readoutHalfBandHz = 1 / (2 * 9.8e-6)
    assert ptone_tx_frequency(hdr, acq, bandPosition=0) == pytest.approx(123_248_104 + readoutHalfBandHz / 2)
    assert ptone_tx_frequency(hdr, acq, bandPosition=1) == pytest.approx(123_248_104 + readoutHalfBandHz)

def test_fov_shift_is_added():
    # A positive readout offset raises the transmit frequency by offset * Hz/mm
    hdr = make_header()
    hzPerMm = (1 / 9.8e-6) / 480.0
    shifted   = ptone_tx_frequency(hdr, make_acq(readoutOffsetMm=10.0))
    unshifted = ptone_tx_frequency(hdr, make_acq())
    assert shifted - unshifted == pytest.approx(10.0 * hzPerMm)

def test_readout_offset_uses_read_dir():
    # Only the component of the position along read_dir counts
    acq = make_acq(readDir=(1.0, 0.0, 0.0))
    acq.position = (5.0, 30.0, -70.0)
    assert readout_frequencies(make_header(), acq)['readoutOffsetMm'] == pytest.approx(5.0)

@pytest.mark.parametrize('side, sign', [('high', 1), ('low', -1)])
@pytest.mark.parametrize('readoutOffsetMm', [-25.0, 0.0, 22.27])
def test_tone_lands_halfway_regardless_of_fov_shift(readoutOffsetMm, side, sign):
    # ptone_readout_offset is the inverse of ptone_tx_frequency: the tone always appears halfway
    # between the imaging band edge and the readout band edge, on the requested side
    hdr, acq = make_header(), make_acq(readoutOffsetMm=readoutOffsetMm)
    halfway = 0.75 / (2 * 9.8e-6)
    assert ptone_readout_offset(ptone_tx_frequency(hdr, acq, side=side), hdr, acq) == pytest.approx(sign * halfway)

@pytest.mark.parametrize('bandPosition', [0, 0.25, 0.5, 1])
@pytest.mark.parametrize('readoutOffsetMm', [-8.37, 0.0, 22.27])
def test_low_side_mirrors_high_side_about_fov_centre(bandPosition, readoutOffsetMm):
    # The low and high side frequencies are symmetric about the centre of the (shifted) FOV
    hdr, acq = make_header(), make_acq(readoutOffsetMm=readoutOffsetMm)
    r = readout_frequencies(hdr, acq)
    fovCentreHz = r['f0Hz'] + r['fovShiftHz']
    high = ptone_tx_frequency(hdr, acq, bandPosition=bandPosition, side='high')
    low  = ptone_tx_frequency(hdr, acq, bandPosition=bandPosition, side='low')
    assert (high - fovCentreHz) == pytest.approx(fovCentreHz - low)
    assert low < fovCentreHz < high

def test_low_side_matches_kstream_formula_without_fov_shift():
    # With 2x oversampling and no FOV shift: f0 - 0.75 * baseResolution * bandwidthPerPixel
    hdr, acq = make_header(), make_acq()
    bwPerPixel = 1 / (512 * 9.8e-6)
    assert ptone_tx_frequency(hdr, acq, side='low') == pytest.approx(123_248_104 - 0.75 * 256 * bwPerPixel)

@pytest.mark.parametrize('bandPosition', [-0.5, -0.01, 1.01, 2])
def test_band_position_outside_0_to_1_is_rejected(bandPosition):
    # Values outside 0-1 would put the tone in the imaging band or outside the readout band
    with pytest.raises(ValueError, match='bandPosition'):
        ptone_tx_frequency(make_header(), make_acq(), bandPosition=bandPosition)

@pytest.mark.parametrize('side', ['HIGH', 'lo', '', None])
def test_unknown_side_is_rejected(side):
    with pytest.raises(ValueError, match='side'):
        ptone_tx_frequency(make_header(), make_acq(), side=side)

# ----- Real datasets --------------------------------------------------------------------------

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataDir = os.environ.get('PTONE_TEST_DATA', os.path.join(repoDir, '..', 'pilot-tone-test-data'))

# Datasets with a known transmit frequency
datasets = [
    pytest.param('ptone20260114b--meas_MID00967_FID47949_t2_tse_tra_dark_fluid_ARIA__nm_pt.h5',
                 123.285661e6,   # Hard-coded in kstream's prot/aria/tx-and-save/t2-tse-tra-dark-fluid.bash
                 id='ptone20260114b'),
    pytest.param('ptoneH20260429--meas_MID00284_FID14131_t2_tse_tra_dark_fluid__m_pt.h5',
                 123.286337e6,   # 123.248129 MHz + 0.75 * 256 * 199 Hz, as in t2-tse-tra-dark-fluid.bash
                 id='ptoneH20260429'),
]

def measure_tone_offset(path, firstLine=100, numLines=400):
    # Header, first imaging acquisition, and the measured pilot tone frequency (Hz from the centre
    # of the readout band): the strongest bin outside the imaging band, summed over lines and channels
    with h5py.File(path, 'r') as f:
        hdr = ismrmrd.xsd.CreateFromDocument(f['dataset']['xml'][0])

    ds = ismrmrd.Dataset(path, 'dataset', create_if_needed=False)
    try:
        acqs = [ds.read_acquisition(i) for i in range(firstLine, min(firstLine + numLines, ds.number_of_acquisitions()))]
    finally:
        ds.close()
    acqs = [a for a in acqs if not a.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)]

    spec = sum(np.abs(np.fft.fftshift(np.fft.fft(a.data, axis=1), axes=1)).sum(axis=0) for a in acqs)
    numSamples = acqs[0].number_of_samples
    dwellS     = acqs[0].sample_time_us * 1e-6
    freqs      = (np.arange(numSamples) - numSamples // 2) / (numSamples * dwellS)
    outer      = np.r_[0:numSamples // 4, 3 * numSamples // 4:numSamples]
    peak       = outer[np.argmax(spec[outer])]
    binWidthHz = 1 / (numSamples * dwellS)
    return hdr, acqs[0], freqs[peak], binWidthHz

@pytest.mark.parametrize('fileName, txFrequencyHz', datasets)
def test_predicted_tone_offset_matches_data(fileName, txFrequencyHz):
    path = os.path.join(dataDir, fileName)
    if not os.path.exists(path):
        pytest.skip('Dataset not found: ' + path)

    hdr, acq, measuredHz, binWidthHz = measure_tone_offset(path)
    predictedHz = ptone_readout_offset(txFrequencyHz, hdr, acq)
    fovShiftHz  = readout_frequencies(hdr, acq)['fovShiftHz']

    # Within one frequency bin of the measured peak
    assert abs(predictedHz - measuredHz) < binWidthHz

    # And the FOV shift correction matters: without it the prediction is more than a bin off
    assert abs((predictedHz + fovShiftHz) - measuredHz) > binWidthHz
