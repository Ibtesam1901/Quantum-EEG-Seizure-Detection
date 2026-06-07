import os
import mne
import numpy as np

# Load EDF

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

# Filter

raw.filter(
    l_freq=0.5,
    h_freq=40.0
)

# First Channel

signal = raw.get_data()[0]

# Hjorth Mobility

first_derivative = np.diff(signal)

mobility = np.sqrt(
    np.var(first_derivative)
    /
    np.var(signal)
)

print("Hjorth Mobility:")
print(mobility)