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
    assert float(args['--tx-delay']) == 0   # Start immediately by default

def test_tx_delay_option(fake_script, tmp_path):
    rf = USRPTransmitter(script_path=str(fake_script), python=sys.executable, tx_delay=0.5)
    try:
        rf.tx(123e6, duration=60)
        args = read_args(tmp_path)
    finally:
        rf.stop()
    assert float(args['--tx-delay']) == pytest.approx(0.5)

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

def test_log_path_gets_output_printed_before_stop(tmp_path):
    # print() output that's still buffered when the transmitter is stopped must reach the log, as
    # with tx_waveforms.py's "Starting to stream waveform..." message
    script = tmp_path / "fake_tx_waveforms_no_flush.py"
    script.write_text('import time\nprint("Starting to stream waveform")\ntime.sleep(30)\n')
    logPath = tmp_path / "usrp.txt"
    rf = USRPTransmitter(script_path=str(script), python=sys.executable, log_path=str(logPath))
    rf.tx(123e6, duration=60)
    end = time.monotonic() + 10
    while "Starting to stream waveform" not in logPath.read_text() and time.monotonic() < end:
        time.sleep(0.05)
    rf.stop()
    assert "Starting to stream waveform" in logPath.read_text()

# Runs multi_usrp_tx() from tx_waveforms.py with uhd.usrp.MultiUSRP replaced by a stand-in, and
# prints the start_time it passes to send_waveform()
START_TIME_CHECK = '''
import sys, types
sys.path.insert(0, SCRIPT_DIR)
import uhd, tx_waveforms

class FakeMultiUSRP:
    def __init__(self, args): pass
    def get_time_now(self): return uhd.types.TimeSpec(100.0)
    def send_waveform(self, data, duration, freq, rate, channels, gain, start_time=None):
        print("start_time:", None if start_time is None else start_time.get_real_secs())

uhd.usrp.MultiUSRP = FakeMultiUSRP
args = types.SimpleNamespace(args="", wave_freq=1000.0, rate=1e6, wave_ampl=0.3, duration=0.01,
                             waveform="sine", freq=123e6, channels=[0], gain=0, tx_delay=TX_DELAY)
tx_waveforms.multi_usrp_tx(args)
'''

@pytest.mark.parametrize('txDelay, expected', [(0.0, 'start_time: None'), (0.5, 'start_time: 100.5')])
def test_tx_waveforms_start_time(txDelay, expected):
    # --tx-delay 0 starts immediately (start_time None); otherwise a timed start txDelay from now
    txPython = os.environ.get('PTONE_TX_PYTHON')
    if not txPython:
        pytest.skip("Set PTONE_TX_PYTHON to a Python interpreter with UHD to run this test")
    scriptDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = START_TIME_CHECK.replace('SCRIPT_DIR', repr(scriptDir)).replace('TX_DELAY', repr(txDelay))
    result = subprocess.run([txPython, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout

def test_log_lines_are_timestamped(tmp_path):
    # Every line in the log starts with the time it was received, in order, and the log ends
    # with the exit code
    import re
    from datetime import datetime
    script = tmp_path / "fake_tx_waveforms_steps.py"
    script.write_text('import time\nprint("step 1")\ntime.sleep(0.3)\nprint("step 2")\ntime.sleep(30)\n')
    logPath = tmp_path / "usrp.txt"
    rf = USRPTransmitter(script_path=str(script), python=sys.executable, log_path=str(logPath))
    rf.tx(123e6, duration=60)
    end = time.monotonic() + 10
    while "step 2" not in logPath.read_text() and time.monotonic() < end:
        time.sleep(0.05)
    rf.stop()

    lines = logPath.read_text().splitlines()
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.*)$')
    assert all(pattern.match(line) for line in lines), lines
    times = [datetime.strptime(pattern.match(line).group(1), '%Y-%m-%d %H:%M:%S,%f') for line in lines]
    assert times == sorted(times)

    messages = [pattern.match(line).group(2) for line in lines]
    assert str(script) in messages[0]                       # The command line
    assert messages[1:3] == ["step 1", "step 2"]            # The output, line by line
    assert messages[-1].startswith("Exited with code")      # Written after stop()
    # The timestamps reflect when each line was printed
    assert (times[2] - times[1]).total_seconds() >= 0.25

def test_log_records_failure_to_start(tmp_path, fake_script):
    logPath = tmp_path / "usrp.txt"
    rf = USRPTransmitter(script_path=str(fake_script), python="/nonexistent/python", log_path=str(logPath))
    with pytest.raises(OSError):
        rf.tx(123e6, duration=60)
    assert "Failed to start" in logPath.read_text().splitlines()[-1]
