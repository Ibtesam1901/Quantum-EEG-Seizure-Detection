import os
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =====================================
# LOAD DATASET
# =====================================

base_dir = os.path.dirname(
    os.path.dirname(__file__)
)

X = np.load(
    os.path.join(
        base_dir,
        "results",
        "X.npy"
    )
)

y = np.load(
    os.path.join(
        base_dir,
        "results",
        "y.npy"
    )
)

print("Original Dataset")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# =====================================
# BALANCE DATASET
# =====================================

seizure_idx = np.where(
    y == 1
)[0]

non_seizure_idx = np.where(
    y == 0
)[0]

np.random.seed(42)

selected_non_seizure = np.random.choice(
    non_seizure_idx,
    size=len(seizure_idx),
    replace=False
)

balanced_idx = np.concatenate([

    seizure_idx,
    selected_non_seizure

])

X_balanced = X[
    balanced_idx
]

y_balanced = y[
    balanced_idx
]

print("\nBalanced Dataset")
print("X Shape:", X_balanced.shape)
print("y Shape:", y_balanced.shape)

print(
    "Seizure Samples:",
    np.sum(y_balanced == 1)
)

print(
    "Non-Seizure Samples:",
    np.sum(y_balanced == 0)
)

# =====================================
# PCA
# =====================================

pca = PCA(
    n_components=4
)

X_pca = pca.fit_transform(
    X_balanced
)

print("\nAfter PCA")
print("Shape:", X_pca.shape)

print(
    "Explained Variance:",
    np.sum(
        pca.explained_variance_ratio_
    )
)

# =====================================
# SCALE FOR ANGLE ENCODING
# =====================================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_scaled = scaler.fit_transform(
    X_pca
)

print("\nScaled Dataset")
print(
    "Min:",
    np.min(X_scaled)
)

print(
    "Max:",
    np.max(X_scaled)
)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y_balanced,

    test_size=0.2,
    random_state=42,
    stratify=y_balanced

)

print("\nTrain Shape")
print(X_train.shape)

print("\nTest Shape")
print(X_test.shape)

# =====================================
# SAVE
# =====================================

results_dir = os.path.join(
    base_dir,
    "results"
)

np.save(
    os.path.join(
        results_dir,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        results_dir,
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        results_dir,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        results_dir,
        "y_test.npy"
    ),
    y_test
)

print("\nFiles Saved")
print("X_train.npy")
print("X_test.npy")
print("y_train.npy")
print("y_test.npy")