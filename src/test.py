"""
test.py
-------
Testing pipeline for the HMM arrhythmia detector.

Steps:
  1. Load a trained HMM model and threshold from disk
  2. Load a test ECG record (healthy or arrhythmic)
  3. Filter the signal and extract RR intervals
  4. Compute log-likelihood under the trained model
  5. Classify as Normal or Possible Arrhythmia
  6. Optionally visualize the results

Usage:
  python src/test.py
  OR called from main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_record
from src.preprocessing import filter_ecg
from src.feature_extraction import extract_rr_sequence
from src.hmm_model import load_model, compute_likelihood, classify
from src.utils import load_threshold_stats, rr_summary
import visualization  # local visualization module (see below)

# ─── Configuration ─────────────────────────────────────────────────────────────

# Test records — mix of normal and abnormal rhythms for evaluation
# MIT-BIH Record Key (some examples):
#   "100" → Mostly normal sinus rhythm
#   "108" → PVCs (Premature Ventricular Contractions)
#   "200" → Frequent PVCs, some abnormal beats
#   "207" → Complete heart block (severe arrhythmia)
#   "214" → Left bundle branch block
TEST_RECORDS = [
    ("100", "Normal - should score high"),
    ("108", "PVCs - should score lower"),
    ("200", "Mixed arrhythmia - should score low"),
]

MODEL_PATH = "models/hmm_model.pkl"
STATS_PATH = "models/threshold_stats.json"


def run_test(record_id, description="", plot=True):
    """
    Run arrhythmia detection on a single ECG record.

    Parameters
    ----------
    record_id : str
        MIT-BIH record ID to test.
    description : str
        Human-readable description for display.
    plot : bool
        Whether to show ECG and RR interval plots.

    Returns
    -------
    result : dict
        Dictionary with keys: record_id, log_likelihood, label
    """

    print("\n" + "─" * 60)
    print(f"  Testing Record: {record_id}  ({description})")
    print("─" * 60)

    # ── Load model and threshold ──────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    model = load_model(MODEL_PATH)
    stats = load_threshold_stats(STATS_PATH)
    threshold = stats["threshold"]

    # ── Load and preprocess ECG ───────────────────────────────────────────────
    signal, r_peaks, fs = load_record(record_id)
    filtered_signal = filter_ecg(signal, fs)

    # ── Extract RR intervals ──────────────────────────────────────────────────
    rr = extract_rr_sequence(r_peaks, fs, filter_outliers=True)

    if len(rr) == 0:
        print(f"[test] ERROR: No valid RR intervals for record {record_id}.")
        return None

    print("\n  RR Interval Summary:")
    rr_summary(rr)

    # ── Compute likelihood ────────────────────────────────────────────────────
    log_likelihood = compute_likelihood(model, rr)
    label = classify(log_likelihood, threshold)

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n  Log-Likelihood (per sample): {log_likelihood:.4f}")
    print(f"  Threshold:                   {threshold:.4f}")
    print(f"  Classification:              {label}")

    # Simple visual indicator
    symbol = "✅" if label == "Normal" else "⚠️"
    print(f"\n  {symbol}  {label.upper()}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    if plot:
        visualization.plot_results(
            signal=filtered_signal,
            r_peaks=r_peaks,
            rr_intervals=rr,
            fs=fs,
            record_id=record_id,
            log_likelihood=log_likelihood,
            threshold=threshold,
            label=label
        )

    return {
        "record_id": record_id,
        "log_likelihood": log_likelihood,
        "threshold": threshold,
        "label": label
    }


def run_all_tests(plot=True):
    """
    Test all records in TEST_RECORDS and print a summary table.

    Parameters
    ----------
    plot : bool
        Whether to display plots for each record.

    Returns
    -------
    results : list of dict
    """
    print("\n" + "=" * 60)
    print("  HMM ARRHYTHMIA DETECTOR — TEST PIPELINE")
    print("=" * 60)

    results = []
    for record_id, description in TEST_RECORDS:
        try:
            result = run_test(record_id, description, plot=plot)
            if result:
                results.append(result)
        except Exception as e:
            print(f"[test] ERROR on record {record_id}: {e}")

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Record':<10} {'Log-Likelihood':<18} {'Label'}")
    print("  " + "-" * 45)
    for r in results:
        flag = "✅" if r["label"] == "Normal" else "⚠️ "
        print(f"  {r['record_id']:<10} {r['log_likelihood']:<18.4f} {flag} {r['label']}")

    return results


if __name__ == "__main__":
    run_all_tests(plot=True)