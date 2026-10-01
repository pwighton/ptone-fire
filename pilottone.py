import ismrmrd
import os
import json
import re
import logging
import threading
import subprocess
import numpy as np
import mrdhelper
from ptone.estimate import analyze_line
from ptone.tx_frequency import check_band_position_and_side, ptone_tx_frequency
from ptone.usrp_transmitter import USRPTransmitter

import sys
import traceback
import time
from datetime import datetime

# Defaults for the outputFolder and outputFileStem parameters in pilottone.json.  Results are saved to
# <outputFolder>/<outputFileStem>--<protocolName>--<YYYYMMDD-HHMMSS-mmm>.npz, timestamped when processing starts
defaultOutputFolder   = "/tmp/ismrmrd-server-output--pilottone"
defaultOutputFileStem = "pilottone"

# Defaults for the pilot tone transmitter parameters in pilottone.json
defaultPtoneTx             = False   # Don't transmit unless the config asks for it
defaultPtoneTxBandPosition = 0.5     # bandPosition in ptone_tx_frequency(): halfway between imaging and readout band edges
defaultPtoneTxSide         = 'high'  # side in ptone_tx_frequency()
defaultPtoneTxDB           = 70      # Transmit gain (dB), as in kstream's prot/aria scripts
defaultPtoneTxMaxDurationS = 3600    # Safety limit on transmission time, in case the transmitter isn't stopped
defaultPtoneTxDeviceArgs   = ''      # UHD device arguments (tx_waveforms.py --args).  Empty: UHD searches for any USRP
# Python that runs ptone/tx_waveforms.py.  It needs UHD, which can't be installed in this environment
# (see ptone/environment-tx.yml), so default to the 'ptone-tx' conda environment alongside this one
defaultPtoneTxPython       = os.path.join(os.path.dirname(sys.prefix), 'ptone-tx', 'bin', 'python')

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
    # When using fast_phase_avg --use-nonimage-scans, ACQ_IS_PARALLEL_CALIBRATION are included
    #'ACQ_IS_PARALLEL_CALIBRATION',            # PATREFSCAN
    # Not in twixtools_mdh.py.  fast_phase_avg.py skips the whole AdjCoilSens measurement instead
    'ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA',
    'ACQ_IS_NAVIGATION_DATA',
]

# Defaults for the dontSkipFlags
# As in twixtools_mdh.py, PATREFANDIMASCAN lines are image lines regardless of other flags
defaultDontSkipFlags = [
    'ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING',  # PATREFANDIMASCAN
]

# Default for the ptoneTxFreqSkipFlags parameter in pilottone.json: lines that can't or wont' be used to calculate
# the pilot tone frequency, and so start the transmitter, because their readout (dwell time, position)
# may differ from the imaging readout.  E.g. the noise line has a 5 us dwell vs 9.8 us in the TSE data.
# This is separate from skipFlags, which only decides which lines are analyzed
defaultPtoneTxFreqSkipFlags = [
    'ACQ_IS_NOISE_MEASUREMENT',
    'ACQ_IS_NAVIGATION_DATA',
    'ACQ_IS_RTFEEDBACK_DATA',
    'ACQ_IS_HPFEEDBACK_DATA',
    'ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA',
    'ACQ_IS_PHASE_STABILIZATION',
    'ACQ_IS_PHASE_STABILIZATION_REFERENCE',
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

def get_git_commit():
    # Git commit of the repository containing this file, noting any uncommitted changes
    repoDir = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(['git', '-C', repoDir, 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        status = subprocess.run(['git', '-C', repoDir, 'status', '--porcelain'],
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 'unknown (%s)' % e
    if status != '':
        commit += ' (with uncommitted changes)'
    return commit

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
    gitCommit = get_git_commit()
    logging.info("Git commit: %s", gitCommit)

    # The pilot tone transmitter is started by the first line the frequency can be calculated from
    # (see ptoneTxFreqSkipFlags), so lines within ptoneTxDelayMs of that line are skipped while it starts up
    ptoneTxDelayMs = mrdhelper.get_json_config_param(config, 'ptoneTxDelayMs', default=0, type='float')
    txStartTimeMs  = None  # Scanner time of the line that started the transmitter
    logging.info("Skipping lines within %g ms of the line that starts the transmitter", ptoneTxDelayMs)

    # Reference channel for relative phase and amplitude (phase_ref_chan_idx in fast_phase_avg.py)
    refChanIdx = mrdhelper.get_json_config_param(config, 'refChanIdx', default=0, type='int')
    logging.info("Using channel %d as the reference channel", refChanIdx)

    # Per-channel midpoint of the phase range, set from the first line after ptoneTxDelayMs.
    # Keeping phases within [midpoint - pi, midpoint + pi] minimizes phase wraps across the scan
    phaseMidpoint = None

    numChanMismatch = 0  # Lines skipped because their channel count differs from the first analyzed line

    transmitter = None  # USRPTransmitter, once transmission has started

    # Settings actually used (config values with defaults filled in), saved with the results
    settings = {
        'gitCommit':      gitCommit,
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

        # Lines that can't be used to calculate the pilot tone frequency and start the transmitter
        ptoneTxFreqSkipFlags, ptoneTxFreqSkipFlagNames = get_flags_config_param(config, 'ptoneTxFreqSkipFlags', defaultPtoneTxFreqSkipFlags)
        logging.info("Not starting the transmitter on lines with flags: %s", ', '.join(ptoneTxFreqSkipFlagNames))
        settings['ptoneTxFreqSkipFlags'] = ptoneTxFreqSkipFlagNames

        # Pilot tone transmitter.  The frequency to broadcast (ptoneTxFreqHz) is ptoneTxOverrideFreqHz if
        # set, otherwise it's calculated by ptone_tx_frequency() from ptoneTxBandPosition and ptoneTxSide.
        # The calculation needs the first imaging line, so ptoneTxFreqHz is set when that arrives
        ptoneTx             = mrdhelper.get_json_config_param(config, 'ptoneTx',             default=defaultPtoneTx,             type='bool')
        ptoneTxBandPosition = mrdhelper.get_json_config_param(config, 'ptoneTxBandPosition', default=defaultPtoneTxBandPosition, type='float')
        ptoneTxSide         = mrdhelper.get_json_config_param(config, 'ptoneTxSide',         default=defaultPtoneTxSide,         type='str')
        ptoneTxDB           = mrdhelper.get_json_config_param(config, 'ptoneTxDB',           default=defaultPtoneTxDB,           type='float')
        ptoneTxOverrideFreqHz = mrdhelper.get_json_config_param(config, 'ptoneTxOverrideFreqHz', default='', type='str')
        ptoneTxOverrideFreqHz = float(ptoneTxOverrideFreqHz) if ptoneTxOverrideFreqHz.strip() != '' else None
        ptoneTxPython       = mrdhelper.get_json_config_param(config, 'ptoneTxPython',       default=defaultPtoneTxPython,       type='str')
        ptoneTxMaxDurationS = mrdhelper.get_json_config_param(config, 'ptoneTxMaxDurationS', default=defaultPtoneTxMaxDurationS, type='float')
        ptoneTxDeviceArgs   = mrdhelper.get_json_config_param(config, 'ptoneTxDeviceArgs',   default=defaultPtoneTxDeviceArgs,   type='str')
        ptoneTxLogPath      = os.path.splitext(outputFilePath)[0] + '--usrp.txt'  # Output of tx_waveforms.py
        check_band_position_and_side(ptoneTxBandPosition, ptoneTxSide)
        ptoneTxFreqHz = None

        if not ptoneTx:
            logging.info("Pilot tone transmitter: off")
        elif ptoneTxOverrideFreqHz is not None:
            logging.info("Pilot tone transmitter: on, %.0f Hz (set by ptoneTxOverrideFreqHz), %g dB", ptoneTxOverrideFreqHz, ptoneTxDB)
        else:
            logging.info("Pilot tone transmitter: on, frequency calculated with bandPosition %g, side '%s', %g dB", ptoneTxBandPosition, ptoneTxSide, ptoneTxDB)
        settings['ptoneTx']             = ptoneTx
        settings['ptoneTxBandPosition'] = ptoneTxBandPosition
        settings['ptoneTxSide']         = ptoneTxSide
        settings['ptoneTxDB']           = ptoneTxDB
        settings['ptoneTxOverrideFreqHz'] = ptoneTxOverrideFreqHz
        settings['ptoneTxFreqHz']         = ptoneTxFreqHz  # Updated at the first imaging line
        settings['ptoneTxPython']         = ptoneTxPython
        settings['ptoneTxDeviceArgs']     = ptoneTxDeviceArgs
        settings['ptoneTxMaxDurationS']   = ptoneTxMaxDurationS
        settings['ptoneTxStarted']        = False          # Updated when transmission starts
        settings['ptoneTxExitCode']       = None           # Set if the transmitter stops before the scan ends
        settings['ptoneTxStartScanCounter'] = None         # Line that started the transmitter (scan_counter)
        settings['ptoneTxStartTimeMs']      = None         # Its scanner timestamp (ms since midnight)

        for item in connection:
            if item is None:
                break

            if not isinstance(item, ismrmrd.Acquisition):
                continue

            # Timestamps are 2.5 ms ticks since midnight
            timeMs = item.acquisition_time_stamp * 2.5

            # The first line the pilot tone frequency can be calculated from starts the transmitter.
            # This happens whether or not the line is analyzed
            if (txStartTimeMs is None) and not any(item.is_flag_set(flag) for flag in ptoneTxFreqSkipFlags):
                txStartTimeMs = timeMs
                settings['ptoneTxStartScanCounter'] = item.scan_counter
                settings['ptoneTxStartTimeMs']      = timeMs
                logging.info("Pilot tone frequency and transmitter start from line scan_counter %d", item.scan_counter)

                if ptoneTxOverrideFreqHz is not None:
                    ptoneTxFreqHz = ptoneTxOverrideFreqHz
                else:
                    try:
                        ptoneTxFreqHz = ptone_tx_frequency(mrdHeader, item, bandPosition=ptoneTxBandPosition, side=ptoneTxSide)
                    except Exception as e:
                        logging.warning("Could not calculate the pilot tone frequency: %s", e)
                if ptoneTxFreqHz is not None:
                    logging.info("Pilot tone frequency: %.0f Hz%s", ptoneTxFreqHz, "" if ptoneTx else " (not transmitted, ptoneTx is false)")
                settings['ptoneTxFreqHz'] = ptoneTxFreqHz

                # Start transmitting.  A failure here is logged but doesn't stop the analysis
                if ptoneTx:
                    if ptoneTxFreqHz is None:
                        logging.error("Pilot tone not transmitted: no frequency")
                    else:
                        try:
                            transmitter = USRPTransmitter(python=ptoneTxPython, device_args=ptoneTxDeviceArgs, log_path=ptoneTxLogPath)
                            transmitter.tx(ptoneTxFreqHz, duration=ptoneTxMaxDurationS, gain=ptoneTxDB)
                            settings['ptoneTxStarted'] = True
                            logging.info("Pilot tone transmitter started: %.0f Hz, %g dB, using %s.  Output in %s",
                                         ptoneTxFreqHz, ptoneTxDB, ptoneTxPython, ptoneTxLogPath)
                        except Exception as e:
                            transmitter = None
                            logging.error("Pilot tone transmitter failed to start: %s", e)

            # Skip lines we don't want to analyze
            if not is_image_line(item, skipFlags, dontSkipFlags):
                continue

            # Skip lines until the pilot tone is on: before the transmitter has started, and within
            # ptoneTxDelayMs of it starting.  The difference is taken modulo one day in case the scan
            # crosses midnight
            if txStartTimeMs is None:
                continue
            if (timeMs - txStartTimeMs) % (24*60*60*1000) < ptoneTxDelayMs:
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
        # Stop transmitting, noting if the transmitter had already stopped (e.g. an error in tx_waveforms.py)
        if transmitter is not None:
            if not transmitter.is_transmitting():
                exitCode = transmitter.wait(timeout=0)
                settings['ptoneTxExitCode'] = exitCode
                logging.warning("Pilot tone transmitter stopped before the end of the scan (exit code %d).  See %s", exitCode, ptoneTxLogPath)
            transmitter.stop()
            logging.info("Pilot tone transmitter stopped")

        if numChanMismatch > 0:
            logging.warning("Skipped %d lines with a mismatched channel count", numChanMismatch)
        save_results(results, outputFilePath, timestamp, config, settings, mrdHeader)
        connection.send_close()
        logging.getLogger().removeHandler(logHandler)
        logHandler.close()

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
             timestamp    = np.array(timestamp),                               # Processing start, YYYYMMDD-HHMMSS-mmm
             config       = np.array(json.dumps(config, indent=4)),            # Config as received, as JSON text
             settings     = np.array(json.dumps(settings, indent=4)),          # Settings actually used, as JSON text
             mrd_header   = np.array(mrd_header_to_xml(mrdHeader)))            # MRD header, as XML text
    logging.info("Saved pilot tone results for %d lines to %s", len(results), filePath)
