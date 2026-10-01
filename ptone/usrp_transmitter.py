#!/usr/bin/env python3
"""
USRP Transmitter class.

Wraps tx_waveforms.py in a subprocess for non-blocking transmission control.

Copied from kstream (kstream/usrp_transmitter.py and kstream/tx_waveforms.py at commit 850c2ea).
Changes from kstream: the Python interpreter that runs tx_waveforms.py is configurable (python),
since it needs the UHD Python bindings, which the environment running this class may not have; and
the subprocess output can be written to a file (log_path, unbuffered) instead of being discarded.

Rename either `tx_waveforms3.15.py` or `tx_waveforms4.py` to `tx_waveforms.py`
to switch vesions 

See: 
  - https://github.com/EttusResearch/uhd/blob/master/host/examples/python/tx_waveforms3.15.py
  - https://github.com/EttusResearch/uhd/blob/master/host/examples/python/tx_waveforms.py
"""

import os
import subprocess
from pathlib import Path
from typing import Optional


class USRPTransmitter:
    """
    Control USRP transmissions via tx_waveforms.py subprocess.
    
    Example:
        rf = USRPTransmitter()
        rf.tx(123.420931e6, duration=10, gain=-10)
        # ... do other things ...
        rf.stop()
    
    With context manager:
        with USRPTransmitter() as rf:
            rf.tx(123.420931e6, duration=10)
            time.sleep(5)
            rf.stop()
    
    With DC at baseband (no offset):
        rf = USRPTransmitter(wave_freq=0)
        rf.tx(123.420931e6, duration=10)
    """
    
    # Default script location: same directory as this module
    _DEFAULT_SCRIPT = Path(__file__).parent / "tx_waveforms.py"
    
    def __init__(
        self,
        script_path: Optional[str] = None,
        device_args: str = "",
        sample_rate: float = 1e6,
        amplitude: float = 0.3,
        wave_freq: float = 1000.0,
        verbose: bool = False,
        python: str = "python3",
        log_path: Optional[str] = None,
    ):
        """
        Initialize the USRP transmitter.
        
        Args:
            script_path: Path to tx_waveforms.py (default: same directory as this module)
            device_args: USRP device arguments (e.g., "serial=34D8B47")
            sample_rate: Sample rate in samples/second (default 1e6)
            amplitude: Waveform amplitude 0-0.7 (default 0.3)
            wave_freq: Baseband tone offset in Hz (default 1000 to avoid LO leakage).
                       If non-zero, the center frequency will be adjusted so
                       that center_freq + wave_freq = desired output frequency.
                       Set to 0 for DC at baseband if preferred.
            verbose: If True, show subprocess output (default False)
            python: Python interpreter used to run tx_waveforms.py.  It needs the UHD Python
                    bindings (import uhd).  Default "python3", found on the PATH
            log_path: If set, subprocess output is appended to this file (overrides verbose)
        """
        self._process: Optional[subprocess.Popen] = None
        
        self.script_path = Path(script_path) if script_path else self._DEFAULT_SCRIPT
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script not found: {self.script_path}")
        
        self.device_args = device_args
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.wave_freq = wave_freq
        self.verbose = verbose
        self.python = python
        self.log_path = log_path
    
    def tx(self, freq: float, duration: float, gain: float = -10.0) -> None:
        """
        Start transmitting a sine wave.
        
        If already transmitting, stops the current transmission first.
        Returns immediately; transmission runs in background.
        
        Args:
            freq: Desired output frequency in Hz (e.g., 123.420931e6)
            duration: Transmission duration in seconds
            gain: TX gain in dB (default -10)
        """
        self.stop()
        
        # Calculate center frequency to account for wave_freq offset
        # Output frequency = center_freq + wave_freq
        # Therefore: center_freq = freq - wave_freq
        center_freq = freq - self.wave_freq
        
        cmd = [
            self.python,
            str(self.script_path),
            "--freq", str(center_freq),
            "--rate", str(self.sample_rate),
            "--wave-freq", str(self.wave_freq),
            "--wave-ampl", str(self.amplitude),
            "--gain", str(int(gain)),
            "--duration", str(duration),
            "--waveform", "sine",
        ]
        
        if self.device_args:
            cmd.extend(["--args", self.device_args])
        
        kwargs = {}
        logFile = None
        if self.log_path is not None:
            logFile = open(self.log_path, "a")
            logFile.write(" ".join(cmd) + "\n")
            logFile.flush()
            kwargs["stdout"] = logFile
            kwargs["stderr"] = subprocess.STDOUT
            # Unbuffered, so print() output reaches the log even when the process is stopped by stop()
            kwargs["env"] = dict(os.environ, PYTHONUNBUFFERED="1")
        elif not self.verbose:
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
        
        try:
            self._process = subprocess.Popen(cmd, **kwargs)
        finally:
            # The subprocess has its own copy of the file handle
            if logFile is not None:
                logFile.close()
    
    def stop(self) -> None:
        """Stop any current transmission."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
    
    def wait(self, timeout: Optional[float] = None) -> int:
        """
        Wait for current transmission to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        
        Returns:
            Return code from subprocess, or -1 if no transmission active
        """
        if self._process is None:
            return -1
        return self._process.wait(timeout=timeout)
    
    def is_transmitting(self) -> bool:
        """Check if currently transmitting."""
        if self._process is None:
            return False
        return self._process.poll() is None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    def __del__(self):
        self.stop()


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Test USRPTransmitter")
    parser.add_argument("--script-path", default=None, help="Path to tx_waveforms.py (default: same directory)")
    parser.add_argument("--freq", type=float, default=123.456e6, help="Frequency in Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--gain", type=float, default=-10.0, help="Gain in dB")
    parser.add_argument("--wave-freq", type=float, default=1000.0, help="Baseband offset in Hz (default: 1000 to avoid LO leakage)")
    parser.add_argument("--verbose", action="store_true", help="Show subprocess output")
    parser.add_argument("--python", default="python3", help="Python interpreter with UHD bindings to run tx_waveforms.py (default: python3)")
    args = parser.parse_args()
    
    print(f"Creating transmitter with wave_freq={args.wave_freq} Hz")
    with USRPTransmitter(
        script_path=args.script_path,
        wave_freq=args.wave_freq,
        verbose=args.verbose,
        python=args.python,
    ) as rf:
        print(f"Starting transmission at {args.freq/1e6:.6f} MHz for {args.duration}s")
        rf.tx(args.freq, args.duration, args.gain)
        
        while rf.is_transmitting():
            print(".", end="", flush=True)
            time.sleep(1)
        
        print("\nDone")
