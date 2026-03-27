"""
hmm_model.py
------------
HMM training and inference for arrhythmia detection, featuring 
a secondary Rule-Based Diagnosis Engine for specific classification.
"""

import numpy as np
import pickle
import json
from hmmlearn.hmm import GaussianHMM

def diagnose_rhythm(rr_sequence, log_likelihood, threshold):
    """
    Secondary classification layer. If the HMM flags an anomaly, 
    this function analyzes the timing characteristics to name the arrhythmia.
    """
    # If the score is above the threshold, it is statistically healthy
    if log_likelihood >= threshold:
        return "Normal Sinus Rhythm"

    # --- ARRHYTHMIA DIAGNOSIS LOGIC ---
    mean_rr = np.mean(rr_sequence)
    bpm = 60.0 / mean_rr
    std_rr = np.std(rr_sequence)
    max_rr = np.max(rr_sequence)

    # 1. Check for Heart Block (Sudden Gaps)
    # If any single beat is > 1.8 seconds, it's a pause/block
    if max_rr > 1.8:
        return "Arrhythmia: Heart Block / Sinus Pause"

    # 2. Check for PVCs (High Variability)
    # If the rhythm is jumpy (high standard deviation)
    if std_rr > 0.12:
        return "Arrhythmia: PVCs (Irregular Rhythm)"

    # 3. Check for Tachycardia (Fast Rate)
    if bpm > 100:
        return "Arrhythmia: Tachycardia (Fast HR)"

    # 4. Check for Bradycardia (Slow Rate)
    if bpm < 50:
        return "Arrhythmia: Bradycardia (Slow HR)"

    # Fallback for nonspecific anomalies
    return "Arrhythmia: Nonspecific Anomaly"


def train_hmm(rr_sequences, n_components=4, n_iter=100, random_state=42):
    """
    Train a Gaussian HMM on a list of healthy RR interval sequences.
    """
    all_rr = []
    lengths = []

    for rr_seq in rr_sequences:
        if len(rr_seq) < n_components:
            continue
        all_rr.append(rr_seq.reshape(-1, 1))
        lengths.append(len(rr_seq))

    if not all_rr:
        raise ValueError("No valid RR sequences available for training.")

    X_train = np.concatenate(all_rr, axis=0)

    # Initialize and train the GaussianHMM
    model = GaussianHMM(
        n_components=n_components,   
        covariance_type="diag",      
        n_iter=n_iter,               
        random_state=random_state,
        verbose=False
    )

    model.fit(X_train, lengths)

    # Compute per-sequence log-likelihoods for thresholding
    train_log_likelihoods = []
    for rr_seq in rr_sequences:
        if len(rr_seq) < n_components:
            continue
        ll = compute_likelihood(model, rr_seq)
        train_log_likelihoods.append(ll)

    return model, train_log_likelihoods


def compute_likelihood(model, rr_sequence):
    """
    Compute normalized log-likelihood of an RR sequence.
    """
    if len(rr_sequence) == 0:
        return -np.inf

    X = rr_sequence.reshape(-1, 1)

    try:
        log_likelihood = model.score(X)
        # Normalize by length for fair comparison
        return log_likelihood / len(rr_sequence)
    except Exception as e:
        return -np.inf


def compute_threshold(train_log_likelihoods, n_std=2.0):
    """
    Compute anomaly threshold: mean - n_std * std.
    """
    lls = np.array(train_log_likelihoods)
    mean_ll = np.mean(lls)
    std_ll = np.std(lls)
    threshold = mean_ll - n_std * std_ll

    return threshold, mean_ll, std_ll


def classify(log_likelihood, threshold):
    """
    Basic classification. (Use diagnose_rhythm for detailed output).
    """
    if log_likelihood >= threshold:
        return "Normal"
    else:
        return "Possible Arrhythmia"


def save_model(model, filepath):
    """Save model to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath):
    """Load model from disk."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model