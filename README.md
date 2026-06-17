# Quantum EEG Seizure Detection using Variational Quantum Classifier (VQC)

## Overview

This project implements a Hybrid Quantum Machine Learning framework for epileptic seizure detection using EEG signals from the CHB-MIT Scalp EEG Database.

The system performs EEG preprocessing, feature extraction, dimensionality reduction, and seizure classification using a Variational Quantum Classifier (VQC) built with Qiskit.

---

## Features

* EEG signal loading from EDF files
* Bandpass filtering (0.5 Hz – 40 Hz)
* 2-second EEG segmentation
* Automatic seizure labeling
* Multi-channel EEG analysis
* Feature extraction using:

  * Wavelet Energy
  * Shannon Entropy
  * Delta Band Power
  * Hjorth Mobility
* Principal Component Analysis (PCA)
* Quantum Feature Encoding
* Variational Quantum Classifier (VQC)
* Performance evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix

---

## Dataset

Dataset: CHB-MIT Scalp EEG Database

Selected Patients:

* CHB01
* CHB03
* CHB05

Selected EDF Files:

### CHB01

* chb01_03.edf
* chb01_04.edf
* chb01_15.edf
* chb01_16.edf

### CHB03

* chb03_01.edf
* chb03_03.edf
* chb03_34.edf

### CHB05

* chb05_06.edf
* chb05_13.edf

Total EDF Files Processed: 9

---

## EEG Channels Used

The following clinically relevant EEG channels were selected:

* FP1-F7
* F3-C3
* FZ-CZ
* P4-O2

---

## Feature Extraction

For each 2-second EEG segment, the following features were extracted from each channel:

1. Wavelet Energy
2. Shannon Entropy
3. Delta Band Power
4. Hjorth Mobility

Total Features:

4 Channels × 4 Features = 16 Features

---

## Dataset Statistics

Generated Dataset:

* Total Segments: 16,191
* Feature Dimensions: 16
* Seizure Segments: 291
* Non-Seizure Segments: 15,900

After Balancing:

* Seizure Samples: 291
* Non-Seizure Samples: 291

Balanced Dataset Size:

582 Samples

---

## Dimensionality Reduction

Principal Component Analysis (PCA) was applied:

* Original Features: 16
* Reduced Features: 4
* Explained Variance: 89.29%

The reduced feature space is used for quantum encoding.

---

## Quantum Model

Quantum Framework:

* Qiskit
* Qiskit Machine Learning

Quantum Components:

### Feature Map

* ZZFeatureMap

### Ansatz

* RealAmplitudes

### Optimizer

* COBYLA

### Classifier

* Variational Quantum Classifier (VQC)

Quantum Configuration:

* 4 Qubits
* Angle Encoding
* Variational Quantum Circuit

---

## Project Structure

```text
Quantum-EEG-Seizure-Detection
│
├── dataset/
│   ├── chb01/
│   ├── chb03/
│   └── chb05/
│
├── preprocessing/
│   ├── load_edf.py
│   ├── segmentation.py
│   ├── filtering.py
│   ├── normalization.py
│   └── labeling.py
│
├── feature_extraction/
│   ├── wavelet_features.py
│   ├── entropy_features.py
│   ├── psd_features.py
│   ├── hjorth_features.py
│   ├── build_full_dataset.py
│   └── check_dataset.py
│
├── quantum_model/
│   ├── prepare_data.py
│   └── vqc_classifier.py
│
├── results/
│   ├── eeg_features.csv
│   ├── X.npy
│   └── y.npy
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Quantum-EEG-Seizure-Detection.git

cd Quantum-EEG-Seizure-Detection
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment:

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Pipeline

### Build Dataset

```bash
python feature_extraction/build_full_dataset.py
```

### Prepare Data

```bash
python quantum_model/prepare_data.py
```

### Train Quantum Classifier

```bash
python quantum_model/vqc_classifier.py
```
## Evaluation Outputs

After training the VQC model, the following files are generated in the `results/` directory:

- `metrics.csv` containing Accuracy, Precision, Recall, and F1 Score.
- `confusion_matrix.png` containing a visualization of the confusion matrix.

These outputs make it easier to analyze and compare model performance.
---
Technologies Used

Python
NumPy
Pandas
SciPy
Scikit-Learn
PyWavelets
MNE
Qiskit
Qiskit Machine Learning
---

Future Enhancements
Streamlit Web Application
Real-Time EEG Processing
Quantum Neural Networks (QNN)
Hyperparameter Optimization
Multi-Class Seizure Classification
Deployment on IBM Quantum Hardware

---

Author

Syed Aakif Zain

Bachelor of Engineering (Computer Science & Engineering)

Quantum EEG Seizure Detection Project
