import os
import mne
import pywt
import numpy as np
import pandas as pd

from scipy.stats import entropy
from scipy.signal import welch

# =====================================
# DATASET CONFIGURATION
# =====================================

FILES = {

    "chb01/chb01_03.edf": (2996, 3036),
    "chb01/chb01_04.edf": (1467, 1494),
    "chb01/chb01_15.edf": (1732, 1772),
    "chb01/chb01_16.edf": (1015, 1066),

    "chb03/chb03_01.edf": (362, 414),
    "chb03/chb03_03.edf": (432, 501),
    "chb03/chb03_34.edf": (1982, 2029),

    "chb05/chb05_06.edf": (417, 532),
    "chb05/chb05_13.edf": (1086, 1196)

}

# =====================================
# CHANNELS
# =====================================

SELECTED_CHANNELS = [

    "FP1-F7",
    "F3-C3",
    "FZ-CZ",
    "P4-O2"

]

# =====================================
# PATHS
# =====================================

base_dir = os.path.dirname(
    os.path.dirname(__file__)
)

dataset_dir = os.path.join(
    base_dir,
    "dataset"
)

results_dir = os.path.join(
    base_dir,
    "results"
)

os.makedirs(
    results_dir,
    exist_ok=True
)

# =====================================
# STORAGE
# =====================================

X = []
y = []

# =====================================
# PROCESS EACH EDF
# =====================================

for edf_file, seizure_range in FILES.items():

    print("\n================================")
    print("Processing:", edf_file)
    print("================================")

    file_path = os.path.join(
        dataset_dir,
        edf_file
    )

    seizure_start = seizure_range[0]
    seizure_end = seizure_range[1]

    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose=False
    )

    raw.filter(
        l_freq=0.5,
        h_freq=40.0,
        verbose=False
    )

    sfreq = int(
        raw.info["sfreq"]
    )

    window_size = 2 * sfreq

    # -----------------------------
    # Get Channel Indices
    # -----------------------------

    channel_indices = []

    for ch in SELECTED_CHANNELS:

        channel_indices.append(
            raw.ch_names.index(ch)
        )

    data = raw.get_data()

    total_segments = 0
    seizure_segments = 0

    signal_length = data.shape[1]

    # -----------------------------
    # Segment Loop
    # -----------------------------

    for start in range(
            0,
            signal_length - window_size,
            window_size):

        end = start + window_size

        segment_start_sec = start / sfreq
        segment_end_sec = end / sfreq

        feature_vector = []

        # -------------------------
        # Extract Features
        # -------------------------

        for idx in channel_indices:

            signal = data[idx,
                          start:end]

            # ---------------------
            # Wavelet Energy
            # ---------------------

            coeffs = pywt.wavedec(
                signal,
                'db4',
                level=4
            )

            wavelet_energy = sum(
                np.sum(c ** 2)
                for c in coeffs
            )

            # ---------------------
            # Shannon Entropy
            # ---------------------

            hist, _ = np.histogram(
                signal,
                bins=50,
                density=True
            )

            hist = hist + 1e-10

            shannon_entropy = entropy(
                hist
            )

            # ---------------------
            # Delta Power
            # ---------------------

            freqs, psd = welch(
                signal,
                fs=sfreq
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

            derivative = np.diff(
                signal
            )

            mobility = np.sqrt(
                np.var(derivative)
                /
                (np.var(signal) + 1e-10)
            )

            feature_vector.extend([

                wavelet_energy,
                shannon_entropy,
                delta_power,
                mobility

            ])

        # -------------------------
        # Label Segment
        # -------------------------

        label = 0

        if (
                segment_start_sec <= seizure_end
                and
                segment_end_sec >= seizure_start
        ):
            label = 1
            seizure_segments += 1

        X.append(
            feature_vector
        )

        y.append(
            label
        )

        total_segments += 1

    print("Segments:", total_segments)
    print("Seizure Segments:", seizure_segments)

# =====================================
# CONVERT TO NUMPY
# =====================================

X = np.array(
    X
)

y = np.array(
    y
)

# =====================================
# SAVE NUMPY
# =====================================

np.save(
    os.path.join(
        results_dir,
        "X.npy"
    ),
    X
)

np.save(
    os.path.join(
        results_dir,
        "y.npy"
    ),
    y
)

# =====================================
# SAVE CSV
# =====================================

columns = []

for ch in SELECTED_CHANNELS:

    columns.extend([

        f"{ch}_energy",
        f"{ch}_entropy",
        f"{ch}_delta_power",
        f"{ch}_mobility"

    ])

df = pd.DataFrame(
    X,
    columns=columns
)

df["label"] = y

df.to_csv(
    os.path.join(
        results_dir,
        "eeg_features.csv"
    ),
    index=False
)

# =====================================
# SUMMARY
# =====================================

print("\n==========================")
print("FINAL DATASET CREATED")
print("==========================")

print("X Shape:", X.shape)
print("y Shape:", y.shape)

print(
    "Seizure Samples:",
    np.sum(y == 1)
)

print(
    "Non-Seizure Samples:",
    np.sum(y == 0)
)

print("\nSaved:")
print("results/X.npy")
print("results/y.npy")
print("results/eeg_features.csv")