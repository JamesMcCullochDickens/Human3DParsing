from typing import Tuple

import numpy as np


def find_bad_values_indices(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    nan_indices = np.where(np.isnan(array))[0]
    inf_indices = np.where(np.isinf(array))[0]
    return nan_indices, inf_indices


def evenly_spaced_samples(arr, k):
    """Returns k evenly spaced samples from the array arr."""
    indices = np.linspace(0, len(arr) - 1, k, dtype=int)  # Sample indices
    return arr[indices]


def check_bad_values_np(array) -> bool:
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if np.isnan(array).any() or np.isinf(array).any():
        nan_indices, inf_indices = find_bad_values_indices(array)
        print("Indices of NaN values:", nan_indices)
        print("Indices of infinity values:", inf_indices)
        return True
    else:
        return False
