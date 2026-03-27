"""
dashboard.py
------------
Final Presentation Version with Fixed UI/CSS. 
Features:
  - High-contrast CSS for Metric visibility.
  - Blind Patient Simulation (No Spoilers).
  - Realistic P-QRS-T Waveform Plotting.
  - Hospital-Style Continuous ECG Monitor.
"""

import os
import sys
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_record
from src.preprocessing import filter_ecg
from src.feature_extraction import extract_rr_sequence
from src.hmm_model import load_model, compute_likelihood, diagnose_rhythm
from src.utils import load_threshold_stats
from src.visualization import plot_ecg_segment, plot_rr_only
from src.simulated_data import generate_random_patient, create_realistic_ecg

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Clinical Cardiac Monitor", page_icon="🫀", layout="wide")

# ─── Custom CSS for UI Visibility ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Force main background and text colors */
    .main { background-color: #0d1117; color: #e6edf3; }
    
    /* Fix Metric Box Visibility */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 1rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-weight: bold !important;
    }
    
    /* Container styling */
    div[data-testid="stMetric"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }

    /* Fix sidebar text */
    .css-1d391kg { color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🏥 Real-Time Clinical ECG Monitor")
st.caption("Autonomous HMM Diagnostic Pipeline for Arrhythmia Detection")
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Input Source")
    source_type = st.radio("System Mode", ["MIT-BIH Database", "Live Patient Simulation"])

    if source_type == "MIT-BIH Database":
        record_id = st.selectbox("Patient Record", ["100", "101", "103", "108", "200", "207"])
    else:
        st.success("🎲 **Autonomous Blind Mode Active**")
        st.info("The system will generate a physiologically possible heartbeat. Diagnosis is hidden until analysis.")

    st.divider()
    ecg_window = st.slider("Monitor Window (sec)", 5, 20, 10)
    run_button = st.button("▶ START MONITORING", use_container_width=True)

# ─── Analysis Execution ───────────────────────────────────────────────────────
MODEL_PATH = "models/hmm_model.pkl"
STATS_PATH = "models/threshold_stats.json"

if not run_button:
    st.info("👈 Configure system mode and press **START MONITORING**.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9e/Sinus_rhythm_labels.svg", width=500)
else:
    # Check for model files
    if not (os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH)):
        st.error("❌ Model not found. Please run `python src/train.py` first.")
        st.stop()

    # Load System Brain
    model = load_model(MODEL_PATH)
    stats = load_threshold_stats(STATS_PATH)
    threshold = stats["threshold"]

    # 1. Data Acquisition
    if source_type == "MIT-BIH Database":
        with st.spinner("Fetching Clinical Data..."):
            signal, r_peaks, fs = load_record(record_id)
            filtered_signal = filter_ecg(signal, fs)
            rr = extract_rr_sequence(r_peaks, fs)
    else:
        with st.spinner("Simulating Live Patient..."):
            # Randomly generated but physiologically possible
            rr, _ = generate_random_patient(n_beats=60)
            fs = 360
            # Realistic P-QRS-T stitching
            filtered_signal, r_peaks = create_realistic_ecg(rr, fs)

    # 2. HMM & Rule-Based Diagnosis
    log_likelihood = compute_likelihood(model, rr)
    diagnosis = diagnose_rhythm(rr, log_likelihood, threshold)

    # 3. Clinical Metrics (Explicitly formatted for visibility)
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Log-Likelihood Score", value=f"{log_likelihood:.3f}", delta=f"{log_likelihood-threshold:.3f}")
    m2.metric(label="Detection Threshold", value=f"{threshold:.3f}")
    m3.metric(label="Calculated HR", value=f"{60.0/np.mean(rr):.1f} BPM")

    # 4. Diagnostic Reveal Banner
    st.write("---")
    if "Normal" in diagnosis:
        st.success(f"### ✅ CLINICAL STATUS: {diagnosis.upper()}")
    else:
        st.error(f"### ⚠️ CLINICAL STATUS: {diagnosis.upper()}")
    st.write("---")

    # 5. The Heartbeat Plot (Hospital Style Continuous Line)
    st.subheader("📺 Real-Time ECG Waveform")
    fig_ecg = plot_ecg_segment(filtered_signal, r_peaks, fs, max_seconds=ecg_window)
    st.pyplot(fig_ecg)

    # 6. Temporal Analysis (RR Bar Graph)
    st.subheader("📊 RR Interval Temporal Analysis")
    fig_rr = plot_rr_only(rr, label="Normal" if "Normal" in diagnosis else "Abnormal")
    st.pyplot(fig_rr)

    # 7. Data Breakdown
    with st.expander("Show Statistical Breakdown"):
        st.table(pd.DataFrame({
            "Metric": ["Mean RR (s)", "Std Deviation (s)", "Min RR (s)", "Max RR (s)"],
            "Value": [f"{np.mean(rr):.3f}", f"{np.std(rr):.3f}", f"{np.min(rr):.3f}", f"{np.max(rr):.3f}"]
        }))