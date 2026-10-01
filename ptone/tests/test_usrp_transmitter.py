# Tests for ptone/usrp_transmitter.py
#
# These don't need a USRP: a stand-in for tx_waveforms.py records its command line and waits.
# test_tx_waveforms_starts additionally checks that the real tx_waveforms.py starts (--help) with
# the Python interpreter in $PTONE_TX_PYTHON, which needs the UHD Python bindings.  It's skipped if
# PTONE_TX_PYTHON isn't set.

import json
import os
import subprocess
import sys
import time

import pytest

from ptone.usrp_transmitter import USRPTransmitter

# Stand-in for tx_waveforms.py: saves its arguments to ARGS_PATH, then waits like a transmission would
FAKE_TX_WAVEFORMS = '''
import json, sys, time
print("fake tx_waveforms started")
with open(ARGS_PATH, "w") as f:
    json.dump(sys.argv[1:], f)
sys.stdout.flush()
time.sleep(30)
'''

@pytest.fixture
def fake_script(tmp_path):
    script = tmp_path / "fake_tx_waveforms.py"
    script.write_text(FAKE_TX_WAVEFORMS.replace('ARGS_PATH', repr(str(tmp_path / "args.json"))))
    return script

def wait_for(path, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return True
        time.sleep(0.05)
    return False

def read_args(tmp_path):
    # The fake script's arguments, as {option: value}
    argsPath = str(tmp_path / "args.json")
    assert wait_for(argsPath), "Fake tx_waveforms.py did not start"
    with open(argsPath) as f:
        args = json.load(f)
    return dict(zip(args[0::2], args[1::2]))

def test_command_line(fake_script, tmp_path):
    rf = USRPTransmitter(script_path=str(fake_script), wave_freq=1000.0, python=sys.executable)
    try:
        rf.tx(123_284_590.0, duration=60, gain=70)
        args = read_args(tmp_path)
    finally:
        rf.stop()
    # The USRP is tuned wave_freq below the requested frequency, so the tone comes out at freq
    assert float(args['--freq']) == pytest.approx(123_284_590.0 - 1000.0)
    assert float(args['--wave-freq']) == pytest.approx(1000.0)
    assert args['--gain'] == '70'
    assert float(args['--duration']) == pytest.approx(60)
    assert args['--waveform'] == 'sine'

def test_is_transmitting_and_stop(fake_script, tmp_path):
    rf = USRPTransmitter(script_path=str(fake_script), python=sys.executable)
    assert not rf.is_transmitting()
    rf.tx(123e6, duration=60)
    read_args(tmp_path)
    assert rf.is_transmitting()
    rf.stop()
    assert not rf.is_transmitting()

def test_context_manager_stops(fake_script, tmp_path):
    with USRPTransmitter(script_path=str(fake_script), python=sys.executable) as rf:
        rf.tx(123e6, duration=60)
        read_args(tmp_path)
        process = rf._process
    assert process.poll() is not None

def test_log_path_gets_command_and_output(fake_script, tmp_path):
    logPath = tmp_path / "usrp.txt"
    rf = USRPTransmitter(script_path=str(fake_script), python=sys.executable, log_path=str(logPath))
    try:
        rf.tx(123e6, duration=60)
        read_args(tmp_path)
        assert wait_for(str(logPath))
        end = time.monotonic() + 10
        while "fake tx_waveforms started" not in logPath.read_text() and time.monotonic() < end:
            time.sleep(0.05)
    finally:
        rf.stop()
    log = logPath.read_text()
    assert str(fake_script) in log.splitlines()[0]   # The command line
    assert "fake tx_waveforms started" in log        # The subprocess output

def test_bad_python_raises(fake_script):
    rf = USRPTransmitter(script_path=str(fake_script), python="/nonexistent/python")
    with pytest.raises(OSError):
        rf.tx(123e6, duration=60)
    assert not rf.is_transmitting()

def test_missing_script_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        USRPTransmitter(script_path=str(tmp_path / "nonexistent.py"))

def test_tx_waveforms_starts():
    txPython = os.environ.get('PTONE_TX_PYTHON')
    if not txPython:
        pytest.skip("Set PTONE_TX_PYTHON to a Python interpreter with UHD to run this test")
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tx_waveforms.py')
    result = subprocess.run([txPython, script, '--help'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--freq' in result.stdout
