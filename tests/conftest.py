from __future__ import annotations

from math import ceil, log2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def generate_euclidean_circuit_results(distances: np.ndarray) -> dict[str, int]:
    results: dict[str, int] = {}
    vecset1_size: int = distances.shape[0]
    vecset2_size: int = distances.shape[1]
    vector_size: int = distances.shape[2]
    vecset1_qubits: int = ceil(log2(vecset1_size))
    vecset2_qubits: int = ceil(log2(vecset2_size))
    data_qubits: int = ceil(log2(vector_size))
    for i in range(vecset1_size):
        partial_index1: str = f"{i:0{vecset1_qubits}b}"
        for j in range(vecset2_size):
            partial_index2: str = partial_index1 + f"{j:0{vecset2_qubits}b}"
            for k in range(vector_size):
                index_string = partial_index2 + f"{k:0{data_qubits}b}1"
                results[index_string] = distances[i][j][k]
    return results
