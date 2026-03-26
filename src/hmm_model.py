"""
hmm_model.py
------------
HMM training and inference for arrhythmia detection.

A Hidden Markov Model (HMM) is a probabilistic model that assumes:
  - There are hidden "states" (e.g., different heart rhythm modes)
  - Each state emits observable values (RR intervals) from a Gaussian distribution
  - The system transitions between states over time

We train a GaussianHMM on healthy ECG data. It learns:
  - State transition probabilities (how likely to stay in or leave a state)
  - Emission distributions (what RR intervals each state tends to produce)

At test time, we compute the log-likelihood of a new RR sequence under this model.
A very low log-likelihood means the sequence is "unusual" → possible arrhythmia.
"""

import numpy as np
import pickle
from hmmlearn.hmm import GaussianHMM


def train_hmm(rr_sequences, n_components=4, n_iter=100, random_state=42):
    """
    Train a Gaussian HMM on a list of healthy RR interval sequences.

    Parameters
    ----------
    rr_sequences : list of np.ndarray
        Each element is a 1D array of RR intervals from one ECG record.
    n_components : int
        Number of hidden states. 3–4 is usually sufficient for ECG.
        More states = more flexible but harder to train.
    n_iter : int
        Maximum EM (Baum-Welch) iterations for fitting.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    model : GaussianHMM
        Trained HMM model.
    train_log_likelihoods : list of float
        Per-sequence log-likelihoods on the training set.
        Used to establish a baseline for anomaly detection threshold.
    """

    # Combine all sequences into one big array for hmmlearn
    # hmmlearn requires a flat 2D array + a lengths list to know where each
    # individual sequence starts and ends
    all_rr = []
    lengths = []

    for rr_seq in rr_sequences:
        if len(rr_seq) < n_components:
            # Skip sequences too short to be meaningful
            print(f"[hmm_model] Skipping sequence of length {len(rr_seq)} "
                  f"(too short for {n_components} states).")
            continue
        all_rr.append(rr_seq.reshape(-1, 1))
        lengths.append(len(rr_seq))

    if not all_rr:
        raise ValueError("No valid RR sequences available for training.")

    X_train = np.concatenate(all_rr, axis=0)

    print(f"[hmm_model] Training HMM: {n_components} states, "
          f"{len(lengths)} sequences, {len(X_train)} total samples.")

    # Initialize and train the GaussianHMM
    model = GaussianHMM(
        n_components=n_components,   # Number of hidden states
        covariance_type="diag",      # Diagonal covariance (faster, stable)
        n_iter=n_iter,               # Max training iterations
        random_state=random_state,
        verbose=False
    )

    model.fit(X_train, lengths)

    print(f"[hmm_model] Training complete. "
          f"Converged: {model.monitor_.converged}")

    # Compute per-sequence log-likelihoods on the training data
    # These will be used to set the anomaly detection threshold
    train_log_likelihoods = []
    for rr_seq in rr_sequences:
        if len(rr_seq) < n_components:
            continue
        ll = compute_likelihood(model, rr_seq)
        train_log_likelihoods.append(ll)

    return model, train_log_likelihoods


def compute_likelihood(model, rr_sequence):
    """
    Compute the log-likelihood of an RR sequence under the trained HMM.

    Higher log-likelihood = sequence fits the model well = likely normal.
    Lower log-likelihood = unusual pattern = possible arrhythmia.

    We normalize by sequence length to make scores comparable across
    sequences of different lengths.

    Parameters
    ----------
    model : GaussianHMM
        Trained HMM model.
    rr_sequence : np.ndarray
        1D array of RR intervals to evaluate.

    Returns
    -------
    log_likelihood_per_sample : float
        Log-likelihood divided by sequence length.
        This normalization allows fair comparison across lengths.
    """
    if len(rr_sequence) == 0:
        return -np.inf

    X = rr_sequence.reshape(-1, 1)

    try:
        log_likelihood = model.score(X)
        # Normalize by length for fair comparison
        return log_likelihood / len(rr_sequence)
    except Exception as e:
        print(f"[hmm_model] WARNING: Could not score sequence: {e}")
        return -np.inf


def compute_threshold(train_log_likelihoods, n_std=2.0):
    """
    Compute an anomaly detection threshold from training log-likelihoods.

    Threshold = mean - n_std * std

    Sequences with likelihood below this threshold are flagged as abnormal.
    Using mean - 2*std covers ~97.7% of the normal distribution.

    Parameters
    ----------
    train_log_likelihoods : list of float
        Log-likelihoods from training records.
    n_std : float
        Number of standard deviations below mean to set threshold.

    Returns
    -------
    threshold : float
        Anomaly detection cutoff value.
    mean_ll : float
        Mean training log-likelihood.
    std_ll : float
        Std of training log-likelihoods.
    """
    lls = np.array(train_log_likelihoods)
    mean_ll = np.mean(lls)
    std_ll = np.std(lls)
    threshold = mean_ll - n_std * std_ll

    print(f"[hmm_model] Threshold: {threshold:.4f} "
          f"(mean={mean_ll:.4f}, std={std_ll:.4f}, n_std={n_std})")

    return threshold, mean_ll, std_ll


def classify(log_likelihood, threshold):
    """
    Classify a sequence as Normal or Abnormal based on likelihood threshold.

    Parameters
    ----------
    log_likelihood : float
        Normalized log-likelihood of the test sequence.
    threshold : float
        Decision boundary computed from training data.

    Returns
    -------
    label : str
        "Normal" or "Possible Arrhythmia"
    """
    if log_likelihood >= threshold:
        return "Normal"
    else:
        return "Possible Arrhythmia"


def save_model(model, filepath):
    """
    Save the trained HMM model to disk using pickle.

    Parameters
    ----------
    model : GaussianHMM
        Trained model to save.
    filepath : str
        Path to .pkl file, e.g., "models/hmm_model.pkl"
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"[hmm_model] Model saved to {filepath}")


def load_model(filepath):
    """
    Load a previously saved HMM model from disk.

    Parameters
    ----------
    filepath : str
        Path to the .pkl file.

    Returns
    -------
    model : GaussianHMM
        Loaded model ready for inference.
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"[hmm_model] Model loaded from {filepath}")
    return model