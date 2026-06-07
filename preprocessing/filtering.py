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
# Before Filtering
# -----------------------

data_before = raw.get_data()

print("Original Shape:")
print(data_before.shape)

# -----------------------
# Bandpass Filter
# -----------------------

raw.filter(
    l_freq=0.5,
    h_freq=40.0
)

# -----------------------
# After Filtering
# -----------------------

data_after = raw.get_data()

print("\nFiltered Shape:")
print(data_after.shape)

print("\nFirst 10 Values Before:")
print(data_before[0][:10])

print("\nFirst 10 Values After:")
print(data_after[0][:10])