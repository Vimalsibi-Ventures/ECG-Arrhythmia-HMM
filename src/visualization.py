"""
visualization.py
----------------
Plotting utilities for ECG signals and RR interval analysis.

Functions here create matplotlib figures showing:
  - The filtered ECG waveform with R-peak markers
  - The RR interval time series
  - A simple result annotation (Normal / Possible Arrhythmia)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_results(signal, r_peaks, rr_intervals, fs,
                 record_id="", log_likelihood=None,
                 threshold=None, label=None,
                 max_seconds=10):
    """
    Create a 2-panel figure:
      - Top: ECG signal with R-peak markers (first `max_seconds` of signal)
      - Bottom: RR interval sequence as a bar chart

    Parameters
    ----------
    signal : np.ndarray
        Filtered ECG signal.
    r_peaks : np.ndarray
        Sample indices of R-peaks.
    rr_intervals : np.ndarray
        RR intervals in seconds.
    fs : float
        Sampling frequency.
    record_id : str
        Record ID for title display.
    log_likelihood : float or None
        Model log-likelihood to display.
    threshold : float or None
        Decision threshold to display.
    label : str or None
        Classification label: "Normal" or "Possible Arrhythmia"
    max_seconds : float
        How many seconds of ECG to show in top panel.
    """

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0d1117")  # Dark background

    # Use GridSpec for flexible layout
    gs = gridspec.GridSpec(2, 1, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Color scheme ──────────────────────────────────────────────────────────
    ecg_color = "#00d4aa"       # Teal for ECG line
    peak_color = "#ff6b6b"      # Red for R-peaks
    rr_color = "#4facfe"        # Blue for RR bars
    normal_color = "#00d4aa"    # Green-ish for normal label
    abnormal_color = "#ff6b6b"  # Red for arrhythmia label
    text_color = "#e6edf3"
    grid_color = "#21262d"

    # ── Panel 1: ECG Signal ───────────────────────────────────────────────────
    ax1.set_facecolor("#161b22")

    # Limit display to first `max_seconds` of signal
    max_samples = int(max_seconds * fs)
    display_signal = signal[:max_samples]
    time_axis = np.arange(len(display_signal)) / fs

    # Plot the filtered ECG trace
    ax1.plot(time_axis, display_signal, color=ecg_color,
             linewidth=0.8, alpha=0.9, label="ECG (filtered)")

    # Mark R-peaks that fall within the displayed window
    visible_peaks = r_peaks[r_peaks < max_samples]
    ax1.scatter(
        visible_peaks / fs,
        display_signal[visible_peaks],
        color=peak_color, s=30, zorder=5,
        label=f"R-peaks ({len(visible_peaks)} shown)"
    )

    ax1.set_title(f"ECG Signal — Record {record_id}",
                  color=text_color, fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Time (seconds)", color=text_color, fontsize=10)
    ax1.set_ylabel("Amplitude (mV)", color=text_color, fontsize=10)
    ax1.tick_params(colors=text_color)
    ax1.spines[:].set_color(grid_color)
    ax1.legend(loc="upper right", fontsize=9,
               facecolor="#21262d", edgecolor=grid_color, labelcolor=text_color)
    ax1.grid(True, color=grid_color, linewidth=0.5, alpha=0.8)

    # ── Panel 2: RR Intervals ─────────────────────────────────────────────────
    ax2.set_facecolor("#161b22")

    beat_indices = np.arange(1, len(rr_intervals) + 1)

    ax2.bar(beat_indices, rr_intervals, color=rr_color,
            alpha=0.75, width=0.7, label="RR Interval")

    # Draw mean RR line
    mean_rr = np.mean(rr_intervals)
    ax2.axhline(mean_rr, color="#ffd700", linewidth=1.5,
                linestyle="--", label=f"Mean RR = {mean_rr:.3f}s")

    ax2.set_title("RR Interval Sequence",
                  color=text_color, fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Beat Index", color=text_color, fontsize=10)
    ax2.set_ylabel("RR Interval (s)", color=text_color, fontsize=10)
    ax2.tick_params(colors=text_color)
    ax2.spines[:].set_color(grid_color)
    ax2.legend(loc="upper right", fontsize=9,
               facecolor="#21262d", edgecolor=grid_color, labelcolor=text_color)
    ax2.grid(True, color=grid_color, linewidth=0.5, alpha=0.8)

    # ── Annotation: Classification Result ────────────────────────────────────
    if label and log_likelihood is not None and threshold is not None:
        result_color = normal_color if label == "Normal" else abnormal_color
        result_text = (
            f"Classification: {label}\n"
            f"Log-Likelihood: {log_likelihood:.4f}  |  "
            f"Threshold: {threshold:.4f}"
        )
        fig.text(
            0.5, 0.01, result_text,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color=result_color,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#21262d",
                      edgecolor=result_color,
                      alpha=0.9)
        )

    plt.suptitle(
        "Real-Time Arrhythmia Detection using HMM",
        color=text_color, fontsize=15, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    plt.show()


def plot_rr_only(rr_intervals, label=None, title="RR Intervals"):
    """
    Minimal plot for RR intervals only.
    Used by the Streamlit dashboard to display just the RR sequence.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds.
    label : str or None
        Classification label for coloring.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    color = "#00d4aa" if label == "Normal" else "#ff6b6b"
    beat_indices = np.arange(1, len(rr_intervals) + 1)

    ax.bar(beat_indices, rr_intervals, color=color, alpha=0.75, width=0.7)
    ax.axhline(np.mean(rr_intervals), color="#ffd700",
               linewidth=1.5, linestyle="--")

    ax.set_title(title, color="#e6edf3", fontsize=11)
    ax.set_xlabel("Beat", color="#e6edf3")
    ax.set_ylabel("RR (s)", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#21262d")
    ax.grid(True, color="#21262d", linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_ecg_segment(signal, r_peaks, fs, max_seconds=10, title="ECG"):
    """
    Plot a segment of the ECG signal with R-peak markers.
    Returns a figure for use in Streamlit.

    Parameters
    ----------
    signal : np.ndarray
        Filtered ECG signal.
    r_peaks : np.ndarray
        R-peak sample indices.
    fs : float
        Sampling frequency.
    max_seconds : float
        How many seconds to show.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    max_samples = int(max_seconds * fs)
    display_signal = signal[:max_samples]
    time_axis = np.arange(len(display_signal)) / fs

    ax.plot(time_axis, display_signal, color="#00d4aa",
            linewidth=0.8, alpha=0.9)

    visible_peaks = r_peaks[r_peaks < max_samples]
    ax.scatter(
        visible_peaks / fs,
        display_signal[visible_peaks],
        color="#ff6b6b", s=25, zorder=5
    )

    ax.set_title(title, color="#e6edf3", fontsize=11)
    ax.set_xlabel("Time (s)", color="#e6edf3")
    ax.set_ylabel("Amplitude (mV)", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#21262d")
    ax.grid(True, color="#21262d", linewidth=0.5)

    plt.tight_layout()
    return fig