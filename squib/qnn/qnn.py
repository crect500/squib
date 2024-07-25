"""Run Quantum Nearest Neighbors."""

from __future__ import annotations

from math import ceil, log2
from typing import TYPE_CHECKING, Iterable

import numpy as np
from qiskit import (
    AncillaRegister,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit_aer import AerSimulator
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from squib.quclidean import quclidean

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

SHOTS16: int = 2**16


class Metrics:

    """
    Store metrics of a model prediction.

    Attributes
    ----------
    confusion_matrix: A matrix containing label predictions vs true values
    accuracy: The percentage of correct predictions

    """

    __slots__ = ("confusion_matrix", "accuracy")

    def __init__(
        self: Metrics, truth: Iterable[int], predictions: Iterable[int],
    ) -> None:
        """
        Populate confusion matrix values.

        Args:
        ----
        truth: Known labels
        predictions: Predicted labels

        """
        self.confusion_matrix = multilabel_confusion_matrix(truth, predictions)
        self.accuracy: float = accuracy_score(truth, predictions)

    def __repr__(self: Metrics) -> str:
        """
        Return all confusion matrix values as one string.

        Return:
        ------
        The confusion matrix contents

        """
        return (
            f"Accuracy: {self.accuracy}\n"
            f"Confusion Matrix\n"
            f"{self.confusion_matrix}"
        )


def preprocess(data: pd.DataFrame) -> pl.DataFrame:
    """
    Convert data into z-scores.

    Args:
    ----
    data: The input array

    Returns:
    -------
    The scaled data

    """
    scaler: StandardScaler = StandardScaler(with_mean=True).set_output(
        transform="polars",
    )
    scaler.fit(data)
    return scaler.transform(data).to_numpy()


def _run_qnn_circuit(
    circuit: QuantumCircuit,
    feature_circuit: QuantumCircuit,
    *,
    backend: None | AerSimulator = None,
    shots: int = SHOTS16,
) -> dict[str, int]:
    """
    Given a base circuit, runs the qnn circuit for one test feature.

    Args:
    ----
    circuit: A quantum circuit with the training set data already encoded
    feature_circuit: A quantum circuit with the test feature encoded
    backend: Backend on which to run the quantum circuit. Default is
    qiskit_aer.AerSimulator
    shots: The number of times to execute the quantum circuit.

    Returns:
    -------
    The results of the quantum circuit execution

    """
    if backend is None:
        backend = AerSimulator()
    qnn_circuit: QuantumCircuit = circuit.compose(feature_circuit)
    qnn_circuit.h(0)
    qnn_circuit.measure_all()
    return (
        backend.run(
            qnn_circuit,
            shots=shots,
        )
        .result()
        .get_counts()
    )


def _assign_label(
    results: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
) -> int:
    """
    Assign result to label based on k-nearest neighbors.

    Args:
    ----
    results: The scaled inner products from processed qnn circuit results
    labels: The labels corresponding to the results
    k: The number of neighbors to check in the k-nearest neighbors algorithm

    Returns:
    -------
    The determined label

    """
    k_indices: np.ndarray[int] = np.argpartition(results, -k)[-k:]
    values, counts = np.unique(labels[k_indices], return_counts=True)
    return values[np.argmax(counts)]


def _execute_qnn(  # noqa: PLR0913
    circuit: QuantumCircuit,
    feature: np.ndarray,
    labels: np.ndarray,
    training_set_size: int,
    vector_size: int,
    *,
    k: int = 3,
    backend: None | AerSimulator = None,
    shots: int = SHOTS16,
) -> int:
    """
    Create and execute a quantum circuit to determine test feature k-nearest neighbor.

    Args:
    ----
    circuit: A quantum circuit with the training set data already encoded
    feature: A quantum circuit with the test feature encoded
    labels: The labels corresponding to the training set features
    training_set_size: The number of features currently in the training set
    vector_size: The cardinality of the feature vectors
    k: The number of neighbors to check in the k-nearest neighbors algorithm
    backend: Backend on which to run the quantum circuit. Default is
    qiskit_aer.AerSimulator
    shots: The number of times to execute the quantum circuit.

    Returns:
    -------
    The determine label for the test feature

    """
    feature_circuit: QuantumCircuit = quclidean.apply_state_to_index(
        0,
        1,
        1,
        feature,
        backend=backend,
        as_circuit=True,
    )
    results: dict[str, int] = _run_qnn_circuit(
        circuit,
        feature_circuit,
        backend=backend,
        shots=shots,
    )
    vector_qubits: int = ceil(log2(len(feature)))
    distances: np.ndarray = quclidean.retrieve_vectors(
        training_set_size,
        1,
        vector_size,
        vector_qubits,
        results,
    )
    return _assign_label(distances[:, 0], labels, k=k)


def run_qnn(  # noqa: PLR0913
    training_set: np.ndarray,
    test_set: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
    backend: None | AerSimulator = None,
    shots: int = SHOTS16,
) -> np.ndarray:
    """
    Label each test feature using quantum circuit implementations of the knn algorithm.

    Args:
    ----
    training_set: The training features
    test_set: The test features to be labeled
    labels: The labels corresponding to the training features
    k: The number of neighbors to check in the k-nearest neighbors algorithm
    backend: Backend on which to run the quantum circuit. Default is
    qiskit_aer.AerSimulator
    shots: The number of times to execute the quantum circuit.

    Returns:
    -------
    The labels for the test features

    """
    if backend is None:
        backend = AerSimulator()
    total_samples: int = len(training_set) + len(test_set)
    address_register_size: int = ceil(log2(total_samples))
    normalized_train_set, normalized_test_set, norm = quclidean.build_unit_vectors(
        training_set,
        test_set,
    )
    vector_size: int = training_set.shape[1]
    data_register_size: int = ceil(log2(normalized_train_set.shape[1]))

    ancillary_register: AncillaRegister = AncillaRegister(1, "a")
    train_address_register: QuantumRegister = QuantumRegister(
        address_register_size,
        "i",
    )
    test_address_register: QuantumRegister = QuantumRegister(1, "j")
    data_register: QuantumRegister = QuantumRegister(data_register_size, "vec")
    cr: ClassicalRegister = ClassicalRegister(
        address_register_size + data_register_size + 2,
    )
    base_circuit: QuantumCircuit = QuantumCircuit(
        ancillary_register,
        train_address_register,
        test_address_register,
        data_register,
        cr,
    )
    base_circuit.h(ancillary_register)
    base_circuit.compose(
        quclidean.encode_vectors(
            normalized_train_set,
            address_register_size,
            control_state=0,
            as_circuit=True,
        ),
        inplace=True,
    )
    for test_feature in normalized_test_set:
        new_label: int = _execute_qnn(
            base_circuit,
            test_feature,
            labels,
            len(normalized_train_set),
            vector_size,
            k=k,
            backend=backend,
            shots=shots,
        )
        normalized_train_set = np.append(
            normalized_train_set,
            np.expand_dims(test_feature, 0),
            axis=0,
        )
        labels = np.append(labels, new_label, axis=0)
        if ceil(log2(len(normalized_train_set))) <= address_register_size:
            base_circuit.compose(
                quclidean.apply_state_to_index(
                    len(normalized_train_set) - 1,
                    address_register_size,
                    0,
                    test_feature,
                    backend=backend,
                    as_circuit=True,
                ),
            )
    return labels


def cross_validate(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
    backend: AerSimulator | None = None,
    shots: int = 2**15,
) -> list[Metrics]:
    """
    Run k-nearest neighbors as a quantum algorithm with Euclidean distance.

    Args:
    ----
    features: The feature data
    labels: The labels corresponding to the feature data
    k: The number of neighbors to check in the k-nearest neighbors algorithm
    backend: Backend on which to run the quantum circuit. Default is
    shots: The number of times to execute the quantum circuit

    """
    if not backend:
        backend: AerSimulator = AerSimulator()
    index_generator: KFold = KFold(shuffle=True)
    metrics: list[Metrics] = []
    for train_index, test_index in index_generator.split(features):
        new_labels: np.ndarray = run_qnn(
            features[train_index],
            features[test_index],
            labels[train_index],
            k=k,
            backend=backend,
            shots=shots,
        )
        metrics.append(Metrics(truth=labels[test_index], predictions=new_labels))

    return metrics
