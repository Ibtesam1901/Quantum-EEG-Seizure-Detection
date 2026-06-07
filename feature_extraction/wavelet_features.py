import os
import mne
import numpy as np
import pywt

# -----------------------
# Load EDF
# -----------------------

base_dir = os.path.dirname(os.path.dirname(__file__))

file_path = os.path.join(
    base_dir,
    "dataset",
    "chb01",
    "chb01_03.edf"
)

raw = mne.io.read_raw_edf(
    file_path,
    preload=True
)

# -----------------------
# Filter
# -----------------------

raw.filter(
    l_freq=0.5,
    h_freq=40.0
)

# -----------------------
# Get EEG Data
# -----------------------

data = raw.get_data()

# Take first channel only
signal = data[0]

# -----------------------
# Wavelet Transform
# -----------------------

coeffs = pywt.wavedec(
    signal,
    'db4',
    level=4
)

# -----------------------
# Wavelet Energy
# -----------------------

energy = []

for coeff in coeffs:
    e = np.sum(coeff ** 2)
    energy.append(e)

print("Wavelet Energy Features:\n")

for i, e in enumerate(energy):
    print(f"Level {i}: {e}")