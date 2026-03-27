"""
train.py
--------
Full training pipeline for the HMM arrhythmia detector.

Steps:
  1. Load healthy ECG records from MIT-BIH
  2. Filter the signals
  3. Extract RR intervals
  4. Train a GaussianHMM
  5. Compute anomaly threshold from training likelihoods
  6. Save model + threshold stats to disk

Usage:
  python src/train.py
  OR called from main.py
"""

import os
import sys

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_multiple_records
from src.preprocessing import filter_ecg
from src.feature_extraction import extract_rr_sequence
from src.hmm_model import train_hmm, compute_threshold, save_model
from src.utils import ensure_dir, save_threshold_stats, rr_summary

# ─── Configuration ─────────────────────────────────────────────────────────────

# MIT-BIH records considered predominantly "normal sinus rhythm"
# These records have mostly normal beats, suitable for training a healthy baseline.
# Full list of MIT-BIH records: 100–234 (not all numbers exist)
HEALTHY_RECORDS = ["100", "101", "103"]

# Where to save model artifacts
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hmm_model.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "threshold_stats.json")

# HMM hyperparameters
N_COMPONENTS = 4      # Number of hidden states
N_ITER = 100          # Maximum EM iterations
N_STD = 2.0           # Threshold = mean - N_STD * std


def run_training():
    """
    Execute the full training pipeline and save artifacts to disk.

    Returns
    -------
    model : GaussianHMM
        Trained HMM model.
    threshold : float
        Anomaly detection threshold.
    """

    print("=" * 60)
    print("  HMM ARRHYTHMIA DETECTOR — TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load ECG records ──────────────────────────────────────────────
    print(f"\n[train] Loading {len(HEALTHY_RECORDS)} healthy ECG records...")
    records = load_multiple_records(HEALTHY_RECORDS)

    if not records:
        raise RuntimeError("No records loaded. Check your internet connection "
                           "or record IDs.")

    # ── Step 2: Preprocess + Extract RR intervals ────────────────────────────
    rr_sequences = []

    for signal, r_peaks, fs in records:
        # Apply bandpass filter to remove noise
        filtered = filter_ecg(signal, fs)

        # Extract RR intervals from annotated R-peaks
        rr = extract_rr_sequence(r_peaks, fs, filter_outliers=True)

        if len(rr) > N_COMPONENTS:
            rr_summary(rr)
            rr_sequences.append(rr)
        else:
            print(f"[train] Skipping: too few RR intervals ({len(rr)}).")

    print(f"\n[train] {len(rr_sequences)} valid RR sequences ready for training.")

    # ── Step 3: Train HMM ────────────────────────────────────────────────────
    print("\n[train] Training Gaussian HMM...")
    model, train_lls = train_hmm(
        rr_sequences,
        n_components=N_COMPONENTS,
        n_iter=N_ITER
    )

    # ── Step 4: Compute anomaly threshold ───────────────────────────────────
    threshold, mean_ll, std_ll = compute_threshold(train_lls, n_std=N_STD)

    # ── Step 5: Save model and stats ─────────────────────────────────────────
    ensure_dir(MODEL_DIR)
    save_model(model, MODEL_PATH)
    save_threshold_stats(STATS_PATH, threshold, mean_ll, std_ll)

    print("\n[train] Training complete!")
    print(f"  Model saved:    {MODEL_PATH}")
    print(f"  Threshold:      {threshold:.4f}")
    print(f"  Stats saved:    {STATS_PATH}")
    print("=" * 60)

    return model, threshold


if __name__ == "__main__":
    run_training()