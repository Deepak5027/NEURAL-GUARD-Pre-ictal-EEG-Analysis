# Pre-seizure prediction — Temporal GNN

A subject-adaptive temporal graph neural network that detects pre-ictal
EEG patterns 5–30 minutes before seizure onset.

## Architecture

```
Raw EEG (23 ch, 256Hz)
  → Bandpass + epoch + normalize      [preprocess.py]
  → PLV-based brain graph             [graph_builder.py]
  → Temporal GNN (STGCN)              [model.py]
  → Focal loss training               [train.py]
  → Patient-specific fine-tuning      [patient_adapter.py]
  → ONNX export → wearable/mobile
```

## Dataset

CHB-MIT Scalp EEG Database (free, PhysioNet)
- 23 pediatric patients
- 916 hours of continuous EEG
- 198 labeled seizures

Download: https://physionet.org/content/chbmit/1.0.0/

## Quickstart

```bash
pip install torch torch-geometric mne scikit-learn onnx onnxruntime

# 1. Preprocess one patient
python preprocess.py          # expects data/chb01/*.edf

# 2. Build graphs + train
python train.py

# 3. Adapt to new patient + export
python patient_adapter.py
```

## Novel contributions

1. Dynamic PLV graph — connectivity graph is rebuilt per epoch,
   capturing how brain network topology changes before seizures.

2. Subject-adaptive fine-tuning — two-phase head-then-full
   fine-tuning with as little as 15 minutes of patient data.

3. Sustained alert logic — requires N consecutive high-probability
   windows before alerting, drastically reducing false alarms.

## File structure

```
seizure_prediction_tgnn/
├── preprocess.py        Stage 1: EEG loading, filtering, epoching
├── graph_builder.py     Stage 2: PLV-based brain graph construction
├── model.py             Stage 3: Temporal GNN (STGCN architecture)
├── train.py             Stage 4: Focal loss training + early stopping
├── patient_adapter.py   Stage 5: Few-shot adaptation + ONNX export
└── README.md
```

## Requirements

```
torch>=2.1
torch-geometric>=2.4
mne>=1.6
scikit-learn>=1.3
numpy>=1.24
onnx>=1.15
onnxruntime>=1.16
scipy>=1.11
```
