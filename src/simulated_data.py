"""
simulated_data.py
----------------
Generates physiologically realistic P-QRS-T waveforms and autonomous 
blind-test rhythm sequences for real-time demonstration.
"""

import numpy as np

def generate_pqrst(fs=360):
    """
    Creates a single synthetic P-QRS-T complex (one heartbeat).
    """
    # Time vector for one complex (~0.6 seconds)
    t = np.linspace(0, 0.6, int(0.6 * fs))
    
    # Model components as gaussians: P, QRS, T
    p_wave = 0.1 * np.exp(-((t - 0.1)**2) / (2 * 0.01**2))
    qrs_complex = 1.0 * np.exp(-((t - 0.2)**2) / (2 * 0.005**2)) # Sharp R-peak
    # Add a small S-dip
    qrs_complex -= 0.2 * np.exp(-((t - 0.22)**2) / (2 * 0.005**2))
    t_wave = 0.2 * np.exp(-((t - 0.45)**2) / (2 * 0.03**2))
    
    return p_wave + qrs_complex + t_wave

def generate_random_patient(n_beats=60):
    """
    Internally selects a rhythm type for a 'Blind Test'.
    """
    modes = ["Normal", "Arrhythmia (Tachycardia)", "Arrhythmia (Bradycardia)", 
             "Arrhythmia (PVCs)", "Arrhythmia (Heart Block)"]
    selected_mode = np.random.choice(modes)
    
    # Logic to create intervals based on mode
    if selected_mode == "Normal":
        rr = np.random.normal(0.8, 0.03, n_beats)
    elif selected_mode == "Arrhythmia (Tachycardia)":
        rr = np.random.normal(0.45, 0.02, n_beats)
    elif selected_mode == "Arrhythmia (Bradycardia)":
        rr = np.random.normal(1.4, 0.05, n_beats)
    elif selected_mode == "Arrhythmia (PVCs)":
        rr = np.random.normal(0.8, 0.03, n_beats)
        for i in np.random.choice(range(1, n_beats-1), int(n_beats*0.15)):
            rr[i], rr[i+1] = 0.45, 1.2
    else: # Heart Block
        rr = np.random.normal(0.9, 0.04, n_beats)
        rr[np.random.choice(range(n_beats), 2)] = 2.2
        
    return np.clip(rr, 0.3, 3.0), selected_mode

def create_realistic_ecg(rr_intervals, fs=360):
    """
    Stitches P-QRS-T complexes together to create a continuous hospital-style wave.
    """
    heartbeat_template = generate_pqrst(fs)
    total_samples = int(np.sum(rr_intervals) * fs) + fs
    ecg_signal = np.zeros(total_samples)
    r_peaks = []
    
    current_idx = 0
    for interval in rr_intervals:
        # Place heartbeat
        end_idx = current_idx + len(heartbeat_template)
        if end_idx >= total_samples: break
        
        ecg_signal[current_idx:end_idx] = heartbeat_template
        r_peaks.append(current_idx + int(0.2 * fs)) # Index of R-peak in template
        
        # Move forward by the RR interval
        current_idx += int(interval * fs)
        
    return ecg_signal, np.array(r_peaks)