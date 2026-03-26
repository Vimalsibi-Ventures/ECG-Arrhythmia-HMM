"""
feature_extraction.py
---------------------
Extracts RR interval sequences from R-peak positions.

RR intervals are the time gaps between consecutive heartbeats.
They are the core feature used in this project because:
  - Normal hearts have fairly regular RR intervals (~0.6–1.0 seconds)
  - Arrhythmias cause abnormal patterns (too fast, too slow, or irregular)

The HMM will learn the normal RR interval patterns and flag deviations.
"""

import numpy as np


def compute_rr_intervals(r_peaks, fs):
    """
    Compute RR intervals from R-peak sample indices.

    RR interval = (sample index of peak[i+1] - sample index of peak[i]) / fs

    This converts from sample differences to time in seconds.

    Parameters
    ----------
    r_peaks : np.ndarray
        Array of R-peak positions (sample indices).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    rr_intervals : np.ndarray
        Array of RR intervals in seconds.
        Length = len(r_peaks) - 1
    """
    if len(r_peaks) < 2:
        print("[feature_extraction] WARNING: Not enough R-peaks to compute RR intervals.")
        return np.array([])

    # Difference between consecutive peak positions, converted to seconds
    rr_intervals = np.diff(r_peaks) / fs

    print(f"[feature_extraction] Computed {len(rr_intervals)} RR intervals. "
          f"Mean={np.mean(rr_intervals):.3f}s, Std={np.std(rr_intervals):.3f}s")

    return rr_intervals


def filter_rr_intervals(rr_intervals, min_rr=0.3, max_rr=2.0):
    """
    Remove physiologically implausible RR intervals.

    Values outside [0.3, 2.0] seconds are likely annotation errors or
    signal artifacts rather than true heartbeats.
      - < 0.3s → heart rate > 200 bpm (extremely rare even in tachycardia)
      - > 2.0s → heart rate < 30 bpm (very severe bradycardia)

    Parameters
    ----------
    rr_intervals : np.ndarray
        Raw RR interval sequence in seconds.
    min_rr : float
        Minimum valid RR interval (default: 0.3s).
    max_rr : float
        Maximum valid RR interval (default: 2.0s).

    Returns
    -------
    filtered : np.ndarray
        Cleaned RR intervals with outliers removed.
    """
    original_len = len(rr_intervals)
    filtered = rr_intervals[(rr_intervals >= min_rr) & (rr_intervals <= max_rr)]

    removed = original_len - len(filtered)
    if removed > 0:
        print(f"[feature_extraction] Removed {removed} outlier RR intervals "
              f"(outside [{min_rr}, {max_rr}]s).")

    return filtered


def extract_rr_sequence(r_peaks, fs, filter_outliers=True):
    """
    Full pipeline: compute and optionally filter RR intervals.

    Parameters
    ----------
    r_peaks : np.ndarray
        R-peak sample positions.
    fs : float
        Sampling frequency.
    filter_outliers : bool
        Whether to remove physiologically impossible values.

    Returns
    -------
    rr : np.ndarray
        Final RR interval sequence in seconds.
    """
    rr = compute_rr_intervals(r_peaks, fs)

    if filter_outliers and len(rr) > 0:
        rr = filter_rr_intervals(rr)

    return rr


def prepare_hmm_input(rr_sequence):
    """
    Reshape an RR interval sequence into the format expected by hmmlearn.

    hmmlearn requires a 2D array of shape (n_samples, n_features).
    Since we use a single feature (RR interval value), n_features = 1.

    Parameters
    ----------
    rr_sequence : np.ndarray
        1D array of RR intervals.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 1)
        Reshaped sequence ready for HMM input.
    lengths : list
        List containing the single length [n_samples].
        Required by hmmlearn when fitting multiple sequences at once.
    """
    X = rr_sequence.reshape(-1, 1)
    lengths = [len(rr_sequence)]
    return X, lengths