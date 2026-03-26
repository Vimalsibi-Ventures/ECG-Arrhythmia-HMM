"""
dashboard.py
------------
Streamlit dashboard for Real-Time Arrhythmia Detection using HMM.

Run with:
  streamlit run app/dashboard.py

Features:
  - Select a MIT-BIH record ID to analyze
  - View the filtered ECG waveform
  - View the RR interval sequence
  - See the HMM likelihood score
  - Get a classification: Normal or Possible Arrhythmia
"""

import os
import sys
import numpy as np
import streamlit as st

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_record
from src.preprocessing import filter_ecg
from src.feature_extraction import extract_rr_sequence
from src.hmm_model import load_model, compute_likelihood, classify
from src.utils import load_threshold_stats
from src.visualization import plot_ecg_segment, plot_rr_only

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arrhythmia Detector — HMM",
    page_icon="🫀",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; color: #e6edf3; }
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-normal {
        color: #00d4aa;
        font-size: 2em;
        font-weight: bold;
        text-align: center;
    }
    .result-abnormal {
        color: #ff6b6b;
        font-size: 2em;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🫀 Real-Time Arrhythmia Detection")
st.caption("Using Hidden Markov Models trained on MIT-BIH Arrhythmia Database")
st.divider()

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Record selector
    record_id = st.selectbox(
        "Select MIT-BIH Record",
        options=["100", "101", "103", "105", "108", "200", "207", "214"],
        help=(
            "100/101/103 → Normal sinus rhythm\n"
            "108 → PVCs (arrhythmia)\n"
            "200 → Mixed arrhythmia\n"
            "207 → Complete heart block"
        )
    )

    ecg_window = st.slider(
        "ECG Window (seconds)",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="How many seconds of ECG to display"
    )

    run_button = st.button("▶ Run Analysis", use_container_width=True)

    st.divider()
    st.markdown("**Model Files Required:**")
    st.code("models/hmm_model.pkl\nmodels/threshold_stats.json")
    st.caption("Run `python src/train.py` first to generate these files.")

# ─── Main Analysis ────────────────────────────────────────────────────────────

MODEL_PATH = "models/hmm_model.pkl"
STATS_PATH = "models/threshold_stats.json"


def check_model_exists():
    return os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH)


if not run_button:
    # Landing state
    st.info("👈 Select a record from the sidebar and click **Run Analysis** to begin.")
    st.markdown("""
    ### How It Works
    1. **Load** — ECG signal fetched from PhysioNet (MIT-BIH database)
    2. **Filter** — Butterworth bandpass filter removes noise (0.5–45 Hz)
    3. **Extract** — RR intervals computed from cardiologist-annotated R-peaks
    4. **Score** — Trained HMM computes log-likelihood of the RR sequence
    5. **Classify** — Compare score against threshold learned from healthy ECGs
    """)

else:
    # ── Check for trained model ───────────────────────────────────────────────
    if not check_model_exists():
        st.error(
            "❌ Trained model not found. "
            "Please run `python src/train.py` first to train the HMM."
        )
        st.stop()

    # ── Load model ────────────────────────────────────────────────────────────
    with st.spinner("Loading model..."):
        model = load_model(MODEL_PATH)
        stats = load_threshold_stats(STATS_PATH)
        threshold = stats["threshold"]
        mean_ll = stats["mean_log_likelihood"]

    # ── Load and process ECG ──────────────────────────────────────────────────
    with st.spinner(f"Downloading record {record_id} from PhysioNet..."):
        try:
            signal, r_peaks, fs = load_record(record_id)
        except Exception as e:
            st.error(f"Could not load record {record_id}: {e}")
            st.stop()

    with st.spinner("Processing ECG signal..."):
        filtered_signal = filter_ecg(signal, fs)
        rr = extract_rr_sequence(r_peaks, fs, filter_outliers=True)

    if len(rr) == 0:
        st.error("No valid RR intervals found in this record.")
        st.stop()

    # ── Compute classification ────────────────────────────────────────────────
    log_likelihood = compute_likelihood(model, rr)
    label = classify(log_likelihood, threshold)

    # ── Display Result ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Log-Likelihood",
            value=f"{log_likelihood:.4f}",
            delta=f"{log_likelihood - threshold:.4f} vs threshold",
            delta_color="normal" if label == "Normal" else "inverse"
        )

    with col2:
        st.metric(
            label="Threshold",
            value=f"{threshold:.4f}",
            help="Mean - 2×Std of training log-likelihoods"
        )

    with col3:
        st.metric(
            label="Mean Heart Rate",
            value=f"{60.0 / np.mean(rr):.1f} BPM",
            help="Based on mean RR interval"
        )

    # ── Classification Banner ─────────────────────────────────────────────────
    st.divider()

    if label == "Normal":
        st.success("✅  NORMAL — No arrhythmia detected. "
                   "RR interval pattern is consistent with the trained healthy model.")
    else:
        st.error("⚠️  POSSIBLE ARRHYTHMIA — RR interval pattern deviates significantly "
                 "from the learned healthy baseline. Manual review recommended.")

    st.divider()

    # ── Plots ─────────────────────────────────────────────────────────────────
    st.subheader("ECG Signal")
    st.caption(f"First {ecg_window}s — Butterworth filtered (0.5–45 Hz) | "
               f"Red dots = R-peaks")
    ecg_fig = plot_ecg_segment(
        filtered_signal, r_peaks, fs,
        max_seconds=ecg_window,
        title=f"Record {record_id} — Filtered ECG"
    )
    st.pyplot(ecg_fig)

    st.subheader("RR Interval Sequence")
    st.caption(f"{len(rr)} RR intervals | Mean = {np.mean(rr):.3f}s | "
               f"Std = {np.std(rr):.3f}s")
    rr_fig = plot_rr_only(rr, label=label, title=f"Record {record_id} — RR Intervals")
    st.pyplot(rr_fig)

    # ── RR Stats Table ─────────────────────────────────────────────────────────
    st.subheader("RR Interval Statistics")
    bpm = 60.0 / rr
    stats_dict = {
        "Metric": ["Count", "Mean RR (s)", "Std RR (s)", "Min RR (s)",
                   "Max RR (s)", "Mean HR (BPM)", "Min HR (BPM)", "Max HR (BPM)"],
        "Value": [
            len(rr),
            f"{np.mean(rr):.3f}",
            f"{np.std(rr):.3f}",
            f"{np.min(rr):.3f}",
            f"{np.max(rr):.3f}",
            f"{np.mean(bpm):.1f}",
            f"{np.min(bpm):.1f}",
            f"{np.max(bpm):.1f}"
        ]
    }
    import pandas as pd
    st.dataframe(
        pd.DataFrame(stats_dict),
        use_container_width=True,
        hide_index=True
    )