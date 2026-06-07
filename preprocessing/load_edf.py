import os
import mne

base_dir = os.path.dirname(os.path.dirname(__file__))

file_path = os.path.join(
    base_dir,
    "dataset",
    "chb01",
    "chb01_03.edf"
)

raw = mne.io.read_raw_edf(file_path, preload=True)

print(raw.info["sfreq"])
print(raw.ch_names)