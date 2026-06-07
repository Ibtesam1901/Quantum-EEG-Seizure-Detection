import os
import mne
import numpy as np

# ---------------------------
# Load EDF File
# ---------------------------

base_dir = os.path.dirname(os.path.dirname(__file__))

file_path = os.path.join(
    base_dir,
    "dataset",
    "chb01",
    "chb01_03.edf"
)

raw = mne.io.read_raw_edf(file_path, preload=True)

data = raw.get_data()

# ---------------------------
# Segmentation Parameters
# ---------------------------

sfreq = int(raw.info["sfreq"])      # 256 Hz
window_size = 2 * sfreq            # 2 seconds = 512 samples

segments = []
labels = []

# ---------------------------
# Seizure Information
# ---------------------------

SEIZURE_START = 2996   # seconds
SEIZURE_END = 3036     # seconds

# ---------------------------
# Create Segments + Labels
# ---------------------------

segment_number = 0

for start in range(
        0,
        data.shape[1] - window_size,
        window_size):

    end = start + window_size

    segment = data[:, start:end]

    # Convert sample index to seconds
    start_sec = start / sfreq
    end_sec = end / sfreq

    # Labeling
    if start_sec < SEIZURE_END and end_sec > SEIZURE_START:
        label = 1      # Seizure
    else:
        label = 0      # Non-Seizure

    segments.append(segment)
    labels.append(label)

    segment_number += 1

# ---------------------------
# Results
# ---------------------------

segments = np.array(segments)
labels = np.array(labels)

print("Total Segments:", len(segments))
print("Segment Shape:", segments[0].shape)

print("\nTotal Seizure Segments:", np.sum(labels == 1))
print("Total Non-Seizure Segments:", np.sum(labels == 0))

import os

base_dir = os.path.dirname(os.path.dirname(__file__))

results_dir = os.path.join(
    base_dir,
    "results"
)

os.makedirs(
    results_dir,
    exist_ok=True
)

np.save(
    os.path.join(results_dir, "y.npy"),
    labels
)

print("\ny saved successfully!")