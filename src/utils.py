"""
utils.py
--------
General-purpose helper functions used across the project.
"""

import numpy as np
import os
import json
import pickle
import logging


# ─── Logging Setup ────────────────────────────────────────────────────────────

def get_logger(name="arrhythmia_hmm"):
    """
    Create and return a simple console logger.

    Parameters
    ----------
    name : str
        Logger name (shows up in log output).

    Returns
    -------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ─── Normalization ─────────────────────────────────────────────────────────────

def normalize_rr(rr_sequence):
    """
    Normalize RR intervals to zero mean and unit variance.

    Normalization helps ensure the HMM's Gaussian emissions
    are well-scaled and not dominated by outliers.

    Parameters
    ----------
    rr_sequence : np.ndarray
        RR interval sequence in seconds.

    Returns
    -------
    normalized : np.ndarray
        Normalized sequence.
    mean : float
        Mean of original sequence (for de-normalization if needed).
    std : float
        Std of original sequence.
    """
    mean = np.mean(rr_sequence)
    std = np.std(rr_sequence)
    if std == 0:
        return rr_sequence - mean, mean, std
    return (rr_sequence - mean) / std, mean, std


# ─── Sliding Window ───────────────────────────────────────────────────────────

def sliding_window(sequence, window_size, step=1):
    """
    Generate overlapping windows from a 1D sequence.

    Useful if you want to score RR intervals in short windows
    rather than all at once (more granular anomaly detection).

    Parameters
    ----------
    sequence : np.ndarray
        Input 1D array.
    window_size : int
        Number of samples per window.
    step : int
        Step size between windows (1 = fully overlapping).

    Yields
    ------
    window : np.ndarray
        Subsequence of length window_size.
    """
    for start in range(0, len(sequence) - window_size + 1, step):
        yield sequence[start : start + window_size]


# ─── File I/O ─────────────────────────────────────────────────────────────────

def save_threshold_stats(filepath, threshold, mean_ll, std_ll):
    """
    Save threshold parameters to a JSON file.

    These stats are needed at test time to classify new sequences.

    Parameters
    ----------
    filepath : str
        Path to save the JSON file.
    threshold : float
    mean_ll : float
    std_ll : float
    """
    stats = {
        "threshold": threshold,
        "mean_log_likelihood": mean_ll,
        "std_log_likelihood": std_ll
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[utils] Threshold stats saved to {filepath}")


def load_threshold_stats(filepath):
    """
    Load threshold parameters from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON file saved by save_threshold_stats().

    Returns
    -------
    stats : dict
        Dictionary with keys: threshold, mean_log_likelihood, std_log_likelihood
    """
    with open(filepath, "r") as f:
        stats = json.load(f)
    print(f"[utils] Threshold stats loaded from {filepath}")
    return stats


def ensure_dir(path):
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


# ─── Summary Statistics ────────────────────────────────────────────────────────

def rr_summary(rr_sequence):
    """
    Print a brief summary of an RR interval sequence.

    Parameters
    ----------
    rr_sequence : np.ndarray
        RR intervals in seconds.
    """
    if len(rr_sequence) == 0:
        print("[utils] RR sequence is empty.")
        return

    bpm = 60.0 / rr_sequence  # Instantaneous heart rate in BPM
    print(f"  RR intervals: n={len(rr_sequence)}, "
          f"mean={np.mean(rr_sequence):.3f}s, "
          f"std={np.std(rr_sequence):.3f}s, "
          f"min={np.min(rr_sequence):.3f}s, "
          f"max={np.max(rr_sequence):.3f}s")
    print(f"  Heart rate:   mean={np.mean(bpm):.1f} BPM, "
          f"min={np.min(bpm):.1f}, max={np.max(bpm):.1f}")