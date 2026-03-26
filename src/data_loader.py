"""
data_loader.py
--------------
Loads ECG signals and annotations from the MIT-BIH Arrhythmia Database
using the `wfdb` library (PhysioNet).

Each MIT-BIH record contains:
  - A signal file (.dat) with raw ECG samples
  - A header file (.hea) with metadata (sampling rate, units, etc.)
  - An annotation file (.atr) with R-peak locations labeled by cardiologists
"""

import wfdb
import numpy as np


def load_record(record_name, db_name="mitdb", channel=0):
    """
    Load an ECG signal and its R-peak annotations from PhysioNet.

    Parameters
    ----------
    record_name : str
        The record ID to load, e.g. "100", "101", "105"
    db_name : str
        PhysioNet database name. Default is "mitdb" (MIT-BIH Arrhythmia DB).
    channel : int
        Which ECG channel to use (0 = MLII lead, the standard choice).

    Returns
    -------
    signal : np.ndarray
        Raw ECG signal as a 1D array of float values.
    r_peaks : np.ndarray
        Sample indices where R-peaks (heartbeats) occur.
    fs : int
        Sampling frequency in Hz (360 Hz for MIT-BIH).
    """

    # Download and read the record from PhysioNet
    # wfdb will cache downloaded files locally
    record = wfdb.rdrecord(record_name, pn_dir=db_name)

    # Read the cardiologist-annotated R-peak positions
    annotation = wfdb.rdann(record_name, "atr", pn_dir=db_name)

    # Extract the single ECG channel as a flat 1D array
    signal = record.p_signal[:, channel]

    # Get sampling frequency (should be 360 Hz for MIT-BIH)
    fs = record.fs

    # R-peak indices — the sample positions of each heartbeat
    r_peaks = annotation.sample

    print(f"[data_loader] Record {record_name}: {len(signal)} samples, "
          f"fs={fs} Hz, {len(r_peaks)} R-peaks found.")

    return signal, r_peaks, fs


def load_multiple_records(record_list, db_name="mitdb", channel=0):
    """
    Load multiple ECG records and return them as a list of tuples.

    Parameters
    ----------
    record_list : list of str
        List of record IDs to load.
    db_name : str
        PhysioNet database name.
    channel : int
        ECG channel to use.

    Returns
    -------
    records : list of (signal, r_peaks, fs)
        Each element is a tuple from load_record().
    """
    records = []
    for rec_id in record_list:
        try:
            data = load_record(rec_id, db_name=db_name, channel=channel)
            records.append(data)
        except Exception as e:
            print(f"[data_loader] WARNING: Could not load record {rec_id}: {e}")
    return records