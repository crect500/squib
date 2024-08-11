from __future__ import annotations

from math import ceil, log2
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.result import Counts
from qiskit_aer import StatevectorSimulator

from squib.qnn import qnn
from squib.quclidean import quclidean
from tests.conftest import generate_euclidean_circuit_results

if TYPE_CHECKING:
    import polars as pl


@given(integers(min_value=1, max_value=100), integers(min_value=1, max_value=100))
def test_preprocess(rows: int, columns: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    data: np.ndarray = random_generator.uniform(size=(rows, columns))
    scaled_data: pl.DataFrame = qnn.preprocess(pd.DataFrame(data))
    assert data.shape == scaled_data.shape


@given(integers(min_value=1, max_value=3), integers(min_value=1, max_value=3))
@settings(deadline=1000, max_examples=5)
def test_run_qnn_circuit(
    vecset1_qubits: int,
    vector_qubits: int,
) -> None:
    backend = StatevectorSimulator()
    random_generator: np.random.Generator = np.random.default_rng()
    vecset1_size: int = int(2**vecset1_qubits)
    vector_size: int = int(2**vector_qubits)
    vecset1: np.ndarray = random_generator.uniform(size=(vecset1_size, vector_size))
    for index, vector in enumerate(vecset1):
        vecset1[index] = vector / np.linalg.norm(vector)
    test_feature: np.ndarray = random_generator.uniform(size=vector_size)
    test_feature = test_feature / np.linalg.norm(test_feature)
    address_register_size: int = ceil(log2(vecset1_size))
    data_register_size: int = ceil(log2(vector_size))
    if address_register_size == 0:
        address_register_size = 1
    if data_register_size == 0:
        data_register_size = 1

    ancillary_register = AncillaRegister(1, "a")
    training_register = QuantumRegister(address_register_size, "i")
    test_register = QuantumRegister(1, "j")
    data_register = QuantumRegister(data_register_size, "vec")
    circuit = QuantumCircuit(
        ancillary_register,
        training_register,
        test_register,
        data_register,
    )
    circuit.h(ancillary_register)
    circuit.h(training_register)
    circuit.compose(
        quclidean.encode_vectors(
            vecset1,
            address_register_size,
            0,
            backend=backend,
            as_circuit=True,
        ),
        [*ancillary_register, *training_register, *data_register],
        inplace=True,
    )
    feature_circuit: QuantumCircuit = quclidean.apply_state_to_index(
        0,
        1,
        1,
        test_feature,
        backend=backend,
        as_circuit=True,
    )
    solution: Counts = qnn._run_qnn_circuit(circuit, feature_circuit)
    assert isinstance(solution, Counts)
    statevector_solution: np.ndarray = qnn._run_qnn_circuit(
        circuit,
        feature_circuit,
        backend=StatevectorSimulator(),
    )
    distances: np.ndarray = quclidean.retrieve_from_statevector(
        vecset1_size,
        1,
        vector_size,
        data_register_size,
        statevector_solution,
    )
    for distance, vector in zip(distances[0], vecset1):
        classical_distance: float = np.sum((vector - test_feature) ** 2) / 2 ** (
            address_register_size + 2
        )
        assert classical_distance == pytest.approx(distance, abs=1e-4)


@given(integers(min_value=7, max_value=50))
def test_assign_label(size: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    values = random_generator.uniform(size=size)
    data: pd.DataFrame = pd.DataFrame(columns=["value", "class"])
    data["value"] = values
    labels: np.ndarray = random_generator.choice(np.arange(2), len(values))
    data["class"] = labels
    new_label, nearest_neighbors = qnn._assign_label(values, labels)
    data: pd.DataFrame = data.sort_values(
        by="value",
        ascending=False,
        ignore_index=True,
    )
    label_counts: pd.DataFrame = data["class"][0:3].value_counts(sort=True)
    assert new_label == label_counts.index[0]


@mock.patch("squib.qnn.qnn._run_qnn_circuit")
@given(integers(min_value=7, max_value=50), integers(min_value=1, max_value=7))
def test_execute_qnn(
    mock_run: mock.Mock,
    train_set_size: int,
    vector_size: int,
) -> None:
    circuit: QuantumCircuit = QuantumCircuit(1)
    random_generator: np.random.Generator = np.random.default_rng()
    labels: np.ndarray = random_generator.choice(np.arange(2), train_set_size)
    random_generator: np.random.Generator = np.random.default_rng()
    feature: np.ndarray = random_generator.uniform(size=vector_size)
    results_array: np.ndarray = random_generator.uniform(
        size=(train_set_size, 1, vector_size),
    )
    results_dict: dict[str, int] = generate_euclidean_circuit_results(results_array)
    mock_run.return_value = results_dict
    for k in [3, 5, 7]:
        with mock.patch("squib.quclidean.quclidean.apply_state_to_index"):
            qnn._execute_qnn(circuit, feature, labels, train_set_size, vector_size, k=k)


@mock.patch("squib.qnn.qnn._execute_qnn")
@given(
    integers(min_value=3, max_value=20),
    integers(min_value=1, max_value=20),
    integers(min_value=1, max_value=7),
)
@settings(deadline=350)
def test_run_circuit(
    mock_execute: mock.Mock,
    train_set_size: int,
    test_set_size: int,
    vector_size: int,
) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    mock_execute.return_value = (
        random_generator.choice(np.arange(2), size=(1,)),
        random_generator.integers(0, 2, size=3),
    )
    training_set: np.ndarray = random_generator.uniform(
        size=(train_set_size, vector_size),
    )
    test_set: np.ndarray = random_generator.uniform(size=(test_set_size, vector_size))
    labels: np.ndarray = random_generator.choice(2, train_set_size)
    with mock.patch("qiskit.QuantumCircuit.compose"), mock.patch(
        "squib.quclidean.quclidean.encode_vectors",
    ):
        new_labels, q_neighbors, k_neighbors = qnn.run_qnn(
            training_set,
            test_set,
            labels,
        )

    assert len(new_labels) == train_set_size + test_set_size


@mock.patch("squib.qnn.qnn.run_qnn")
@mock.patch("squib.qnn.qnn.multilabel_confusion_matrix")
@mock.patch("squib.qnn.qnn.accuracy_score")
@given(integers(min_value=10, max_value=50))
def test_cross_validate(
    mock_accuracy: mock.Mock,
    mock_confusion: mock.Mock,
    mock_run: mock.Mock,
    feature_quantity: int,
) -> None:
    fold_quantity: int = 5
    random_generator: np.random.Generator = np.random.default_rng()
    fake_labels: np.ndarray = random_generator.choice(
        [0, 1],
        size=ceil(feature_quantity / fold_quantity),
    )
    mock_run.return_value = fake_labels
    mock_confusion.return_value = random_generator.integers(0, 1, size=(2, 2))
    mock_accuracy.return_value = random_generator.integers(0, 1, size=len(fake_labels))
    features: np.ndarray = random_generator.uniform(size=(feature_quantity, 2))
    labels: np.ndarray = random_generator.choice([0, 1], feature_quantity)
    metrics: list[qnn.Metrics] = qnn.cross_validate(features, labels)
    assert len(metrics) == fold_quantity
