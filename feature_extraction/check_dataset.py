import os
import numpy as np

base_dir = os.path.dirname(os.path.dirname(__file__))

X = np.load(
    os.path.join(base_dir, "results", "X.npy")
)

y = np.load(
    os.path.join(base_dir, "results", "y.npy")
)

print("X Shape:", X.shape)
print("y Shape:", y.shape)

print("\nFirst Feature Vector:")
print(X[0])

print("\nFirst Label:")
print(y[0])

print("\nSeizure Samples:")
print(sum(y == 1))

print("\nNon-Seizure Samples:")
print(sum(y == 0))