import os
import mne
import numpy as np

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
# Bandpass Filter
# -----------------------

raw.filter(
    l_freq=0.5,
    h_freq=40.0
)

# -----------------------
# Get EEG Data
# -----------------------

data = raw.get_data()

print("Original Shape:")
print(data.shape)

# -----------------------
# Z-Score Normalization
# -----------------------

normalized_data = (
    data - np.mean(data, axis=1, keepdims=True)
) / np.std(data, axis=1, keepdims=True)

# -----------------------
# Results
# -----------------------

print("\nNormalized Shape:")
print(normalized_data.shape)

print("\nMean of First Channel:")
print(np.mean(normalized_data[0]))

print("\nStd of First Channel:")
print(np.std(normalized_data[0]))

print("\nFirst 10 Normalized Values:")
print(normalized_data[0][:10])