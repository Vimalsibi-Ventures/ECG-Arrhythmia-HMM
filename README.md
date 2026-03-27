# 🫀 Clinical-Grade Arrhythmia Detection using HMM

This project implements a hybrid **Hidden Markov Model (HMM)** and **Rule-Based Diagnostic Engine** for the real-time detection and classification of cardiac arrhythmias. By analyzing **RR intervals** from the MIT-BIH Arrhythmia Database, the system learns the statistical "rhythm" of a healthy heart and flags deviations with specific clinical labels.

---

## 📌 Problem Statement

Traditional heart rate monitors rely on simple thresholds (e.g., HR > 100 BPM), which fail to detect complex irregular rhythms like Premature Ventricular Contractions (PVCs) or Heart Blocks that may occur at a "normal" average rate. This project uses **probabilistic modeling** to understand the sequential nature of heartbeats, providing a more robust detection mechanism.

---

## 🏗️ System Architecture

```
[ Input Stream ] ───► MIT-BIH Database OR Live Patient Simulation
             │
             ▼
    [ Preprocessing ] ────► 4th-order Butterworth Bandpass (0.5–45 Hz)
             │              (Removes baseline wander & electrical noise)
             ▼
 [ Feature Extraction ] ───► RR Interval Calculation (Seconds)
             │              Physiological Outlier Filtering (0.3s - 2.0s)
             ▼
    [ HMM Engine ] ────────► GaussianHMM (4 Hidden States)
             │              Calculates Log-Likelihood of the sequence
             ▼
 [ Diagnosis Engine ] ─────► Hybrid Statistical & Rule-Based Logic
             │              Classifies: Normal, Tachycardia, Bradycardia, PVCs, Block
             ▼
    [ Visualization ] ─────► Live Hospital-Style ECG Monitor (Streamlit)
```

---

## 🚀 Key Features

### 1. Autonomous "Blind" Simulation
The system includes a **Live Patient Simulation** mode that generates physiologically possible P-QRS-T waveforms. It internally selects a random cardiac condition, allowing for true "blind testing" of the diagnostic logic.

### 2. Multi-Class Diagnostic Engine
Unlike basic detectors, this system provides specific clinical labels:
* **Normal Sinus Rhythm**: Stable intervals within learned bounds.
* **Tachycardia/Bradycardia**: Consistent fast or slow rates.
* **PVCs**: Identifying ectopic beats and compensatory pauses.
* **Heart Block**: Detecting sudden sinus pauses exceeding 1.8 seconds.

### 3. Hospital-Style Visualization
The Streamlit dashboard features a high-contrast, real-time ECG waveform plot that "stitches" together synthetic heartbeat templates to mimic a professional bedside monitor.

---

## 🚀 Methodology

### 1. Data Acquisition
* ECG records are streamed directly from **PhysioNet** using the `wfdb` library.
* Training is performed on "healthy" records (e.g., `100`, `101`, `103`).

### 2. Signal Cleaning
* A **4th-order Butterworth bandpass filter (0.5–45 Hz)** is applied.
* Removes baseline wander and high-frequency noise.
* Ensures clean and sharp R-peak detection.

### 3. Feature Extraction
* Detect R-peaks using annotations.
* Compute **RR intervals** (time difference between consecutive R-peaks).
* These sequences act as observations for the HMM.

### 4. HMM Training
* Model: **Gaussian Hidden Markov Model**.
* Hidden States: `4`.
* Learns the distribution of normal RR intervals and transition probabilities between heart states.

### 5. Inference & Classification
* New ECG sequences are evaluated using **log-likelihood scoring**.
* **Classification rule**: If Log-Likelihood < (Mean - 2 × Std Dev) → Possible Arrhythmia; Else → Normal.

---

## 📂 Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/vimalsibi-ventures/ecg-arrhythmia-hmm.git
cd ecg-arrhythmia-hmm
```

### Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# OR
.\venv\Scripts\activate       # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 How to Run

### 1. Train the Model
Generate the baseline model and anomaly detection threshold:
```bash
python src/train.py
```

**Outputs**:
* `models/hmm_model.pkl`
* `models/threshold_stats.json`

---

### 2. Run Test Pipeline
Evaluate the model against arrhythmia records (e.g., Record 108, 200):
```bash
python main.py
```

**Optional Flags**:
* `--no-plot` → Run without visualization
* `--test-only` → Skip training if model already exists

---

### 3. Launch Dashboard
For an interactive real-time visualization:
```bash
streamlit run app/dashboard.py
```

---

## 📊 Diagnostic Outcomes

| Feature | Significance |
| :--- | :--- |
| **Hidden States (4)** | Represents internal cardiac timing modes (Steady, Short, Long, Transition). |
| **Log-Likelihood** | A statistical "fit score"; lower scores indicate higher probability of arrhythmia. |
| **Zero-Phase Filter** | Ensures R-peak timing accuracy by eliminating phase distortion. |
| **P-QRS-T Generator** | Provides a realistic visual trace for hardware-ready simulation. |

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 🙌 Acknowledgements

* PhysioNet MIT-BIH Arrhythmia Database.
* `wfdb` Python Library.
* `hmmlearn` for HMM implementation.
* `Streamlit` for dashboard visualization.