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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit.library import EfficientSU2

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
# FULL DATASET USED
# =====================================

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =====================================
# CLASSICAL BASELINE
# =====================================

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("\n--- CLASSICAL BASELINE (Random Forest) ---")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Precision:", precision_score(y_test, rf_preds))
print("Recall:", recall_score(y_test, rf_preds))
print("F1 Score:", f1_score(y_test, rf_preds))

svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_preds = svc.predict(X_test)
print("\n--- CLASSICAL BASELINE (SVC) ---")
print("Accuracy:", accuracy_score(y_test, svc_preds))
print("Precision:", precision_score(y_test, svc_preds))
print("Recall:", recall_score(y_test, svc_preds))
print("F1 Score:", f1_score(y_test, svc_preds))

# =====================================
# FEATURE MAP
# =====================================

feature_map = PauliFeatureMap(
    feature_dimension=4,
    reps=2,
    paulis=['Z', 'ZZ']
)

# =====================================
# ANSATZ
# =====================================

ansatz = EfficientSU2(
    num_qubits=4,
    reps=3,
    entanglement='linear'
)

# =====================================
# OPTIMIZER
# =====================================

optimizer = COBYLA(
    maxiter=400
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