# 🫀 Real-Time Arrhythmia Detection using HMM

This project implements a modular, simulation-based pipeline for detecting cardiac arrhythmias using **Hidden Markov Models (HMM)**. By analyzing **RR intervals** (the time between heartbeats) from the MIT-BIH Arrhythmia Database, the system learns the statistical "rhythm" of a healthy heart and flags deviations as potential arrhythmias.

---

## 📌 Problem Statement

Traditional threshold-based heart rate monitoring often misses complex irregular rhythms. This project uses a **probabilistic approach (HMM)** to model the sequential nature of heartbeats, enabling more robust detection of anomalies such as:

- Premature Ventricular Contractions (PVCs)
- Heart Blocks
- Irregular rhythm transitions

---

## 🏗️ System Architecture

```
[ Raw ECG Signal ] (PhysioNet/mitdb)
             │
             ▼
    [ Preprocessing ] ────► Butterworth Bandpass Filter (0.5–45 Hz)
             │              (Removes baseline wander & noise)
             ▼
 [ Feature Extraction ] ───► R-Peak Detection (wfdb annotations)
             │              Extract RR Interval Sequences (seconds)
             ▼
    [ HMM Engine ] ────────► GaussianHMM (Hidden States: 4)
             │              Learns Normal Sinus Rhythm transitions
             ▼
  [ Anomaly Detector ] ────► Log-Likelihood Scoring
             │              Threshold = Mean - 2 × Std
             ▼
    [ Visualization ] ─────► Streamlit Dashboard / Matplotlib
```

---

## 🚀 Methodology

### 1. Data Acquisition
- ECG records are streamed directly from **PhysioNet** using the `wfdb` library.
- Training is performed on "healthy" records (e.g., `100`, `101`).

### 2. Signal Cleaning
- A **4th-order Butterworth bandpass filter (0.5–45 Hz)** is applied.
- Removes:
  - Baseline wander  
  - High-frequency noise  
- Ensures clean and sharp R-peak detection.

### 3. Feature Extraction
- Detect R-peaks using annotations.
- Compute **RR intervals** (time difference between consecutive R-peaks).
- These sequences act as observations for the HMM.

### 4. HMM Training
- Model: **Gaussian Hidden Markov Model**
- Hidden States: `4`
- Learns:
  - Distribution of normal RR intervals  
  - Transition probabilities between heart states  

### 5. Inference & Classification
- New ECG sequences are evaluated using **log-likelihood scoring**.
- Classification rule:

```
If Log-Likelihood < (Mean - 2 × Std Dev) → Possible Arrhythmia
Else → Normal
```

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

**Outputs:**
- `models/hmm_model.pkl`
- `models/threshold_stats.json`

---

### 2. Run Test Pipeline
Evaluate the model against arrhythmia records (e.g., Record 108, 200):

```bash
python main.py
```

**Optional Flags:**
- `--no-plot` → Run without visualization  
- `--test-only` → Skip training if model already exists  

---

### 3. Launch Dashboard
For an interactive real-time visualization:

```bash
streamlit run app/dashboard.py
```

---

## 📊 Sample Output

| Record | Type           | Log-Likelihood | Result                 |
|--------|----------------|----------------|------------------------|
| 100    | Normal Sinus   | -1.1245        | ✅ Normal              |
| 108    | PVCs           | -5.8921        | ⚠️ Possible Arrhythmia |
| 207    | Heart Block    | -12.453        | ⚠️ Possible Arrhythmia |

---

## 💡 Key Highlights

- Probabilistic modeling of sequential heart activity  
- More robust than threshold-based detection systems  
- Modular pipeline for easy experimentation and upgrades  
- Real-time visualization with Streamlit  
- Scalable for future deep learning integration  

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 🙌 Acknowledgements

- PhysioNet MIT-BIH Arrhythmia Database  
- `wfdb` Python Library  
- `hmmlearn` for HMM implementation  
- `Streamlit` for dashboard visualization  