import numpy as np

QUASI_FLOAT = 1_000_000


def convert_to_quasi_float(f: float) -> np.int32:
    return np.int32(f * QUASI_FLOAT)


def convert_from_quasi_float(q: np.int32) -> float:
    return float(q) / QUASI_FLOAT
