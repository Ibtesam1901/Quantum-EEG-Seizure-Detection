import os
import mne
import numpy as np
import pywt

from scipy.stats import entropy
from scipy.signal import welch

# -------------------------
# Load EDF
# -------------------------

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

# -------------------------
# Filter
# -------------------------

raw.filter(
    l_freq=0.5,
    h_freq=40.0
)

# -------------------------
# Get Data
# -------------------------

data = raw.get_data()

# Use first channel only
signal = data[0]

# -------------------------
# Segmentation
# -------------------------

sfreq = int(raw.info["sfreq"])

window_size = 2 * sfreq

feature_matrix = []

# -------------------------
# Feature Extraction
# -------------------------

for start in range(
        0,
        len(signal) - window_size,
        window_size):

    segment = signal[start:start + window_size]

    # ---------------------
    # Wavelet Energy
    # ---------------------

    coeffs = pywt.wavedec(
        segment,
        'db4',
        level=4
    )

    wavelet_energy = np.sum(
        coeffs[0] ** 2
    )

    # ---------------------
    # Shannon Entropy
    # ---------------------

    hist, _ = np.histogram(
        segment,
        bins=50,
        density=True
    )

    hist = hist + 1e-10

    shannon_entropy = entropy(hist)

    # ---------------------
    # PSD
    # ---------------------

    freqs, psd = welch(
        segment,
        fs=256
    )

    delta_mask = (
            (freqs >= 0.5)
            &
            (freqs <= 4)
    )

    delta_power = np.sum(
        psd[delta_mask]
    )

    # ---------------------
    # Hjorth Mobility
    # ---------------------

    first_derivative = np.diff(
        segment
    )

    mobility = np.sqrt(
        np.var(first_derivative)
        /
        np.var(segment)
    )

    # ---------------------
    # Feature Vector
    # ---------------------

    feature_vector = [
        wavelet_energy,
        shannon_entropy,
        delta_power,
        mobility
    ]

    feature_matrix.append(
        feature_vector
    )

# -------------------------
# Convert to NumPy
# -------------------------

X = np.array(
    feature_matrix
)

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nFirst Feature Vector:")
print(X[0])

results_dir = os.path.join(base_dir, "results")

os.makedirs(results_dir, exist_ok=True)

np.save(
    os.path.join(results_dir, "X.npy"),
    X
)
print("\nX saved successfully!")