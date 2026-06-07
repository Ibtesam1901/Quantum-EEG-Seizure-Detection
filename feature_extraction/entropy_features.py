import os
import mne
import numpy as np
from scipy.stats import entropy

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

# Create histogram probabilities
hist, _ = np.histogram(
    signal,
    bins=100,
    density=True
)

hist = hist + 1e-10

shannon_entropy = entropy(hist)

print("Shannon Entropy:")
print(shannon_entropy)