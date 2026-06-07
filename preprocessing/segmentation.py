import os
import mne
import numpy as np

base_dir = os.path.dirname(os.path.dirname(__file__))

file_path = os.path.join(
    base_dir,
    "dataset",
    "chb01",
    "chb01_03.edf"
)

raw = mne.io.read_raw_edf(file_path, preload=True)

data = raw.get_data()

sfreq = int(raw.info["sfreq"])

window_size = 2 * sfreq

segments = []

for start in range(
        0,
        data.shape[1] - window_size,
        window_size):

    segment = data[:, start:start + window_size]

    segments.append(segment)

print("Number of Segments:", len(segments))

print("Shape of First Segment:")

print(segments[0].shape)
