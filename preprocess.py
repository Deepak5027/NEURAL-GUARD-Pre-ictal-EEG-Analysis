"""
Stage 1 — EEG Preprocessing
Loads CHB-MIT .edf files, bandpass filters, removes artifacts,
and returns fixed-length epochs ready for graph construction.
"""

import numpy as np
import mne
from pathlib import Path
from scipy.signal import butter, sosfilt

SFREQ       = 256          # Hz — CHB-MIT native sample rate
EPOCH_SEC   = 30           # seconds per window
OVERLAP_SEC = 15           # 50% overlap for sliding window
BAND_LOW    = 0.5          # Hz
BAND_HIGH   = 40.0         # Hz
N_CHANNELS  = 23           # standard CHB-MIT montage


def load_edf(edf_path: str) -> mne.io.Raw:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    return raw


def bandpass(data: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, data, axis=-1)


def remove_artifacts(data: np.ndarray, threshold: float = 150e-6) -> np.ndarray:
    """Zero out epochs where any channel exceeds amplitude threshold (150 µV)."""
    peak = np.max(np.abs(data), axis=-1, keepdims=True)   # (C, 1)
    mask = (peak < threshold).astype(np.float32)
    return data * mask


def epoch_signal(data: np.ndarray, fs: float,
                 epoch_sec: float, overlap_sec: float) -> np.ndarray:
    """
    Slide a window over data.
    Returns: (N_epochs, N_channels, N_samples)
    """
    step   = int((epoch_sec - overlap_sec) * fs)
    length = int(epoch_sec * fs)
    starts = range(0, data.shape[-1] - length + 1, step)
    return np.stack([data[:, s:s + length] for s in starts], axis=0)


def normalize(epochs: np.ndarray) -> np.ndarray:
    """Z-score each channel independently across time."""
    mu  = epochs.mean(axis=-1, keepdims=True)
    std = epochs.std(axis=-1, keepdims=True) + 1e-8
    return (epochs - mu) / std


def preprocess_edf(edf_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for one .edf file.
    Returns: (N_epochs, N_channels, N_samples)  float32
    """
    raw  = load_edf(edf_path)
    data = raw.get_data()                                  # (C, T)  Volts
    data = bandpass(data, BAND_LOW, BAND_HIGH, SFREQ)
    data = remove_artifacts(data)
    epochs = epoch_signal(data, SFREQ, EPOCH_SEC, OVERLAP_SEC)
    epochs = normalize(epochs)
    return epochs.astype(np.float32)


if __name__ == "__main__":
    sample = Path("data/chb01/chb01_01.edf")
    if sample.exists():
        X = preprocess_edf(str(sample))
        print(f"Preprocessed shape: {X.shape}")   # e.g. (59, 23, 7680)
    else:
        print("Place CHB-MIT .edf files under data/chb01/")
