import os
import numpy as np
np.random.seed(42)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit_machine_learning.algorithms import VQC

from qiskit_algorithms.optimizers import COBYLA

# =====================================
# LOAD DATA
# =====================================

base_dir = os.path.dirname(
    os.path.dirname(__file__)
)

X_train = np.load(
    os.path.join(
        base_dir,
        "results",
        "X_train.npy"
    )
)

X_test = np.load(
    os.path.join(
        base_dir,
        "results",
        "X_test.npy"
    )
)

y_train = np.load(
    os.path.join(
        base_dir,
        "results",
        "y_train.npy"
    )
)

y_test = np.load(
    os.path.join(
        base_dir,
        "results",
        "y_test.npy"
    )
)

# =====================================
# SMALL SUBSET
# =====================================

X_train = X_train[:200]
y_train = y_train[:200]

X_test = X_test[:50]
y_test = y_test[:50]

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =====================================
# FEATURE MAP
# =====================================

feature_map = ZZFeatureMap(
    feature_dimension=4,
    reps=2
)

# =====================================
# ANSATZ
# =====================================

ansatz = RealAmplitudes(
    num_qubits=4,
    reps=2
)

# =====================================
# OPTIMIZER
# =====================================

optimizer = COBYLA(
    maxiter=100
)

# =====================================
# VQC
# =====================================

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)

print("\nTraining VQC...\n")

vqc.fit(
    X_train,
    y_train
)

print("Training Completed")

# =====================================
# PREDICTION
# =====================================

predictions = vqc.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(accuracy)
print("\nAccuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))