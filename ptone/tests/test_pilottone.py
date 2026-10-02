# Tests for pilottone.py's output naming
#
# The real-header tests look for data in $PTONE_TEST_DATA (default ../pilot-tone-test-data) and
# ../20260930-bay1-tests relative to the repository root, and are skipped if not found.

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repoDir)
import pilottone

dataDir = os.environ.get('PTONE_TEST_DATA', os.path.join(repoDir, '..', 'pilot-tone-test-data'))

def header_with_measurement_id(measurementID):
    return SimpleNamespace(measurementInformation=SimpleNamespace(measurementID=measurementID, protocolName='test_protocol'))

@pytest.mark.parametrize('measurementID, mid', [
    ('45407_0000026092514332225800000049_0000026092514332225800000049_1104', 1104),
    ('67026_0000026042812130895600000021_0000026042812130895600000021_284', 284),
    ('284', 284),
    ('45407_abc_def', None),
    ('', None),
    (None, None),
])
def test_get_mid(measurementID, mid):
    assert pilottone.get_mid(header_with_measurement_id(measurementID)) == mid

@pytest.mark.parametrize('header', [None, 'not valid MRD XML', SimpleNamespace()])
def test_get_mid_without_header(header):
    assert pilottone.get_mid(header) is None

@pytest.mark.parametrize('path, mid', [
    (os.path.join(dataDir, 'ptoneH20260429--meas_MID00284_FID14131_t2_tse_tra_dark_fluid__m_pt.h5'), 284),
    (os.path.join(repoDir, '..', '20260930-bay1-tests', 'bay1-test1.h5'), 1104),   # Header from the FIRE stream
])
def test_get_mid_real_headers(path, mid):
    if not os.path.exists(path):
        pytest.skip('Not found: ' + path)
    import h5py, ismrmrd
    with h5py.File(path, 'r') as f:
        header = ismrmrd.xsd.CreateFromDocument(f['dataset']['xml'][0])
    assert pilottone.get_mid(header) == mid

class FakeConnection(list):
    def send_close(self):
        pass

def run_process(tmp_path, header):
    import ismrmrd
    acqs = FakeConnection()
    for i in range(5):
        data = np.exp(2j * np.pi * 0.3 * np.arange(256))[None, :].repeat(4, 0) + 0.01 * np.random.randn(4, 256)
        acq = ismrmrd.Acquisition.from_array(data.astype(np.complex64))
        acq.scan_counter = i + 1
        acq.acquisition_time_stamp = 4 * i
        acqs.append(acq)
    pilottone.process(acqs, {'parameters': {'outputFolder': str(tmp_path), 'outputFileStem': 'sub01', 'ptonePlot': 'false'}}, header)
    return sorted(os.listdir(tmp_path))

def test_output_filename_has_mid(tmp_path):
    files = run_process(tmp_path, header_with_measurement_id('45407_x_y_1104'))
    assert [f.rsplit('--', 1)[0] for f in files] == ['sub01--MID01104-test_protocol'] * 2
    assert {os.path.splitext(f)[1] for f in files} == {'.npz', '.txt'}

def test_output_filename_without_mid(tmp_path):
    files = run_process(tmp_path, None)
    assert all(f.startswith('sub01--MIDunknown-unknown--') for f in files)
