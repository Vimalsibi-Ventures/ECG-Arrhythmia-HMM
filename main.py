"""
main.py
-------
End-to-end pipeline for HMM-based arrhythmia detection.

This script:
  1. Trains the HMM on healthy ECG records (or loads a saved model)
  2. Tests on a set of ECG records
  3. Prints results

Usage:
  python main.py              # Train + Test
  python main.py --test-only  # Skip training, use saved model
  python main.py --train-only # Train only, skip testing
"""

import os
import sys
import argparse

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import load_record
from src.preprocessing import filter_ecg
from src.feature_extraction import extract_rr_sequence
from src.hmm_model import load_model, compute_likelihood, classify
from src.utils import load_threshold_stats, rr_summary, ensure_dir

# ─── CLI Arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="HMM-based ECG Arrhythmia Detector"
)
parser.add_argument(
    "--test-only",
    action="store_true",
    help="Skip training; use the saved model in models/"
)
parser.add_argument(
    "--train-only",
    action="store_true",
    help="Train the model and exit without testing"
)
parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Disable matplotlib visualization"
)
args = parser.parse_args()

MODEL_PATH = "models/hmm_model.pkl"
STATS_PATH = "models/threshold_stats.json"

# Records to test
TEST_RECORDS = [
    ("100", "Expected: Normal"),
    ("108", "Expected: Arrhythmia (PVCs)"),
    ("200", "Expected: Arrhythmia (mixed)"),
]


def print_banner():
    print("\n" + "═" * 65)
    print("   🫀  Real-Time Arrhythmia Detection using Hidden Markov Models")
    print("       MIT-BIH Arrhythmia Database  |  PhysioNet")
    print("═" * 65 + "\n")


def main():
    print_banner()
    ensure_dir("models")

    # ── TRAINING PHASE ─────────────────────────────────────────────────────────
    if not args.test_only:
        from src.train import run_training
        model, threshold = run_training()
    else:
        print("[main] Skipping training (--test-only flag set).")
        if not os.path.exists(MODEL_PATH):
            print("[main] ERROR: No saved model found. "
                  "Run without --test-only first.")
            sys.exit(1)

    if args.train_only:
        print("[main] Training complete. Exiting (--train-only flag set).")
        return

    # ── TESTING PHASE ──────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("   TESTING PHASE")
    print("─" * 65)

    # Load model and threshold from disk
    model = load_model(MODEL_PATH)
    stats = load_threshold_stats(STATS_PATH)
    threshold = stats["threshold"]

    results = []
    plot = not args.no_plot

    for record_id, description in TEST_RECORDS:
        print(f"\n→ Record {record_id}  ({description})")

        try:
            # Load ECG
            signal, r_peaks, fs = load_record(record_id)

            # Preprocess
            filtered = filter_ecg(signal, fs)

            # Extract RR intervals
            rr = extract_rr_sequence(r_peaks, fs, filter_outliers=True)

            if len(rr) == 0:
                print(f"  WARNING: No valid RR intervals. Skipping.")
                continue

            rr_summary(rr)

            # Score under HMM
            log_likelihood = compute_likelihood(model, rr)
            label = classify(log_likelihood, threshold)

            # Print
            flag = "✅" if label == "Normal" else "⚠️ "
            print(f"\n  Log-Likelihood : {log_likelihood:.4f}")
            print(f"  Threshold      : {threshold:.4f}")
            print(f"  Result         : {flag} {label}")

            # Plot
            if plot:
                from src.visualization import plot_results
                plot_results(
                    signal=filtered,
                    r_peaks=r_peaks,
                    rr_intervals=rr,
                    fs=fs,
                    record_id=record_id,
                    log_likelihood=log_likelihood,
                    threshold=threshold,
                    label=label
                )

            results.append((record_id, log_likelihood, label))

        except Exception as e:
            print(f"  ERROR: {e}")

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("   FINAL RESULTS")
    print("═" * 65)
    print(f"  {'Record':<10} {'Log-Likelihood':<18} {'Label'}")
    print("  " + "─" * 40)
    for record_id, ll, label in results:
        flag = "✅" if label == "Normal" else "⚠️ "
        print(f"  {record_id:<10} {ll:<18.4f} {flag} {label}")
    print()


if __name__ == "__main__":
    main()