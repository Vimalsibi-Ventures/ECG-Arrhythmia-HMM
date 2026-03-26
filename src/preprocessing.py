"""
preprocessing.py
----------------
Signal preprocessing for ECG data.

Raw ECG signals contain noise from:
  - Baseline wander (very low frequency drift, <0.5 Hz)
  - Muscle artifacts and powerline interference (>45 Hz)

A Butterworth bandpass filter removes these by keeping only
the physiologically meaningful range: 0.5 Hz to 45 Hz.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order. Higher = sharper cutoff, but more phase distortion.

    Returns
    -------
    b, a : filter coefficients
    """
    # Nyquist frequency is half the sampling rate
    nyq = 0.5 * fs

    # Normalize cutoff frequencies to [0, 1] range (relative to Nyquist)
    low = lowcut / nyq
    high = highcut / nyq

    # Design the Butterworth filter
    b, a = butter(order, [low, high], btype="band")
    return b, a


def filter_ecg(signal, fs, lowcut=0.5, highcut=45.0, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter to an ECG signal.

    Using filtfilt (forward + backward pass) ensures zero phase distortion,
    which is important for preserving R-peak timing accuracy.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal (1D array).
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Lower cutoff frequency (default: 0.5 Hz, removes baseline wander).
    highcut : float
        Upper cutoff frequency (default: 45 Hz, removes high-freq noise).
    order : int
        Butterworth filter order (default: 4).

    Returns
    -------
    filtered_signal : np.ndarray
        Filtered ECG signal, same shape as input.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)

    # filtfilt applies the filter twice (forward + backward) for zero phase
    filtered_signal = filtfilt(b, a, signal)

    print(f"[preprocessing] ECG filtered: {lowcut}–{highcut} Hz bandpass applied.")
    return filtered_signal


def normalize_signal(signal):
    """
    Normalize an ECG signal to zero mean and unit variance.

    This is useful for consistent plotting and model input,
    though it's not strictly required for RR interval extraction.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    normalized : np.ndarray
        Signal with mean=0 and std=1.
    """
    mean = np.mean(signal)
    std = np.std(signal)

    # Avoid division by zero for flat signals
    if std == 0:
        return signal - mean

    return (signal - mean) / std