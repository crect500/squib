"""Run Quantum Nearest Neighbors."""

from __future__ import annotations

import logging
from copy import deepcopy
from math import ceil, log2
from typing import TYPE_CHECKING

import numpy as np
from qiskit import (
    AncillaRegister,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit_aer import AerSimulator, StatevectorSimulator
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from squib.evaluation.knn import assign_label
from squib.evaluation.metrics import Metrics
from squib.quclidean import quclidean

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from qiskit.result import Counts

SHOTS16: int = 2**16
logger: logging.Logger = logging.getLogger(__name__)


def preprocess(data: pd.DataFrame) -> np.ndarray:
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


def _run_qknn_circuit(
    circuit: QuantumCircuit,
    feature_circuit: QuantumCircuit,
    *,
    backend: None | AerSimulator = None,
    shots: int = SHOTS16,
) -> Counts | np.ndarray:
    """
    Given a base circuit, runs the QKNN circuit for one test feature.

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
    qknn_circuit: QuantumCircuit = circuit.compose(
        feature_circuit,
        [*circuit.qregs[0], *circuit.qregs[2], *circuit.qregs[3]],
    )
    qknn_circuit.h(0)
    measurement_bits = ClassicalRegister(qknn_circuit.num_qubits)
    if isinstance(backend, StatevectorSimulator):
        return np.real(backend.run(qknn_circuit).result().get_statevector())

    qknn_circuit.add_register(measurement_bits)
    qknn_circuit.measure_all(add_bits=False)
    return (
        backend.run(
            qknn_circuit,
            shots=shots,
        )
        .result()
        .get_counts()
    )


def _execute_qknn(  # noqa: PLR0913
    circuit: QuantumCircuit,
    feature: np.ndarray,
    labels: np.ndarray,
    training_set_size: int,
    vector_size: int,
    *,
    k: int = 3,
    backend: None | AerSimulator = None,
    shots: int = SHOTS16,
) -> tuple[int, np.ndarray[int]]:
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
    if backend is None:
        backend = AerSimulator()
    feature_circuit: QuantumCircuit = quclidean.apply_state_to_index(
        0,
        1,
        1,
        feature,
        backend=backend,
        as_circuit=True,
    )
    results: dict[str, int] | np.ndarray = _run_qknn_circuit(
        circuit,
        feature_circuit,
        backend=backend,
        shots=shots,
    )
    vector_qubits: int = ceil(log2(len(feature)))
    if vector_qubits == 0:
        vector_qubits = 1
    if isinstance(backend, StatevectorSimulator):
        distances = quclidean.retrieve_from_statevector(
            training_set_size,
            1,
            vector_size,
            vector_qubits,
            results,
        )
    else:
        distances: np.ndarray = quclidean.retrieve_vectors(
            training_set_size,
            1,
            vector_size,
            vector_qubits,
            results,
        )

    return assign_label(distances[:, 0], labels, k=k)


def run_qknn(  # noqa: PLR0913
    training_set: np.ndarray,
    test_set: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
    backend: None | AerSimulator = None,
    shots: int = SHOTS16,
) -> tuple[np.ndarray[int], np.ndarray[int], np.ndarray[int]]:
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
    The labels for the test features and the sorted q- and k-nearest neighbors

    """
    if backend is None:
        backend = AerSimulator()
    working_labels: np.ndarray = deepcopy(labels)
    total_samples: int = len(training_set) + len(test_set)
    address_register_size: int = ceil(log2(total_samples))
    normalized_train_set, normalized_test_set, _ = quclidean.build_unit_vectors(
        training_set,
        test_set,
    )
    vector_size: int = training_set.shape[1]
    data_register_size: int = ceil(log2(normalized_train_set.shape[1]))
    if data_register_size == 0:
        data_register_size = 1

    ancillary_register = AncillaRegister(1, "a")
    train_address_register = QuantumRegister(
        address_register_size,
        "i",
    )
    test_address_register = QuantumRegister(1, "j")
    data_register = QuantumRegister(data_register_size, "vec")
    if not isinstance(backend, StatevectorSimulator):
        cr = ClassicalRegister(
            address_register_size + data_register_size + 2,
        )
        base_circuit = QuantumCircuit(
            ancillary_register,
            train_address_register,
            test_address_register,
            data_register,
            cr,
        )
    else:
        base_circuit = QuantumCircuit(
            ancillary_register,
            train_address_register,
            test_address_register,
            data_register,
        )
    base_circuit.h(ancillary_register)
    base_circuit.h(train_address_register)
    base_circuit.compose(
        quclidean.encode_vectors(
            normalized_train_set,
            address_register_size,
            control_state=0,
            backend=backend,
            as_circuit=True,
        ),
        [*ancillary_register, *train_address_register, *data_register],
        inplace=True,
    )
    new_labels: np.ndarray = np.ndarray((len(normalized_test_set),), dtype=int)
    q_nearest_neighbors: np.ndarray[int] = np.ndarray(
        (len(normalized_test_set), k),
        dtype=int,
    )
    k_nearest_neighbors: np.ndarray[int] = np.ndarray(
        (len(normalized_test_set), k),
        dtype=int,
    )
    for iteration, test_feature in enumerate(normalized_test_set):
        logger.warning(f"Building circuit {iteration + 1} / {len(normalized_test_set)}")
        new_label, q_greatest = _execute_qknn(
            base_circuit,
            test_feature,
            working_labels,
            len(normalized_train_set),
            vector_size,
            k=k,
            backend=backend,
            shots=shots,
        )
        new_labels[iteration] = new_label
        q_nearest_neighbors[iteration] = q_greatest
        elementwise_difference: np.ndarray[float] = np.subtract(
            normalized_train_set,
            test_feature,
        )
        normalized_train_set = np.append(
            normalized_train_set,
            np.expand_dims(test_feature, 0),
            axis=0,
        )
        k_nearest_neighbors[iteration] = assign_label(
            np.sum(elementwise_difference * elementwise_difference, axis=1),
            labels=working_labels,
            k=k,
        )[1]
        working_labels = np.append(working_labels, new_label)
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
                [*ancillary_register, *train_address_register, *data_register],
                inplace=True,
            )
    return new_labels, q_nearest_neighbors, k_nearest_neighbors


def cross_validate(  # noqa: PLR0913
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
    backend: AerSimulator | StatevectorSimulator | None = None,
    shots: int = SHOTS16,
    seed: int = 1,
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
    seed: Seed for the cross validation

    """
    if not backend:
        backend: AerSimulator = AerSimulator()
    index_generator: StratifiedKFold = StratifiedKFold(shuffle=True, random_state=seed)
    metrics: list[Metrics] = []
    for iteration, (train_index, test_index) in enumerate(
        index_generator.split(features, labels),
        start=1,
    ):
        logger.warning(f"Training fold {iteration} / 5")
        new_labels, q_nearest_neighbors, k_nearest_neighbors = run_qknn(
            features[train_index],
            features[test_index],
            labels[train_index],
            k=k,
            backend=backend,
            shots=shots,
        )
        logger.warning(f"Truth {labels[test_index]}")
        logger.warning(f"Predictions {new_labels}")
        metrics.append(
            Metrics(
                truth=labels[test_index],
                predictions=new_labels,
                quantum_neighbors=q_nearest_neighbors,
                classical_neighbors=k_nearest_neighbors,
            ),
        )
        logger.warning(metrics[-1])

    return metrics
