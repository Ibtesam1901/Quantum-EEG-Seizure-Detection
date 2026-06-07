import os
import mne
import numpy as np
from scipy.signal import welch

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

# First channel

signal = raw.get_data()[0]

# PSD using Welch

freqs, psd = welch(
    signal,
    fs=256
)

# Delta Band (0.5–4 Hz)

delta_mask = (freqs >= 0.5) & (freqs <= 4)

delta_power = np.sum(
    psd[delta_mask]
)

print("Delta Band Power:")
print(delta_power)