from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis.strategies import integers
from qiskit import QuantumCircuit

from squib.qnn import qnn
from tests.conftest import generate_euclidean_circuit_results

if TYPE_CHECKING:
    import polars as pl


@given(integers(min_value=1, max_value=100), integers(min_value=1, max_value=100))
def test_preprocess(rows: int, columns: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    data: np.ndarray = random_generator.uniform(size=(rows, columns))
    scaled_data: pl.DataFrame = qnn.preprocess(pd.DataFrame(data))
    assert data.shape == scaled_data.shape


@mock.patch("qiskit.result.Result.get_counts")
@given(
    integers(min_value=1, max_value=50),
    integers(min_value=1, max_value=50),
    integers(min_value=1, max_value=7),
)
@settings(deadline=300)
def test_run_qnn_circuit(
    mock_counts: mock.Mock,
    vecset1_size: int,
    vecset2_size: int,
    vector_size: int,
) -> None:
    circuit: QuantumCircuit = QuantumCircuit(1)
    feature_circuit: QuantumCircuit = QuantumCircuit(1)
    result_array: np.ndarray = np.ndarray((vecset1_size, vecset2_size, vector_size))
    results_dict: dict[str, int] = generate_euclidean_circuit_results(result_array)
    mock_counts.return_value = results_dict
    solution: dict[str, int] = qnn._run_qnn_circuit(circuit, feature_circuit)
    assert results_dict == solution


@given(integers(min_value=7, max_value=50))
def test_assign_label(size: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    values = random_generator.uniform(size=size)
    data: pd.DataFrame = pd.DataFrame(columns=["value", "class"])
    data["value"] = values
    labels: np.ndarray = random_generator.choice(np.arange(2), len(values))
    data["class"] = labels
    new_label: int = qnn._assign_label(values, labels)
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
    integers(min_value=1, max_value=20),
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
    mock_execute.return_value = random_generator.choice(np.arange(2), size=(1,))
    training_set: np.ndarray = random_generator.uniform(
        size=(train_set_size, vector_size),
    )
    test_set: np.ndarray = random_generator.uniform(size=(test_set_size, vector_size))
    labels: np.ndarray = random_generator.choice(2, train_set_size)
    with mock.patch("qiskit.QuantumCircuit.compose"), mock.patch(
        "squib.quclidean.quclidean.encode_vectors",
    ):
        new_labels: np.ndarray = qnn.run_qnn(training_set, test_set, labels)

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
        [0, 1], size=ceil(feature_quantity / fold_quantity),
    )
    mock_run.return_value = fake_labels
    mock_confusion.return_value = random_generator.integers(0, 1, size=(2, 2))
    mock_accuracy.return_value = random_generator.integers(0, 1, size=len(fake_labels))
    features: np.ndarray = random_generator.uniform(size=(feature_quantity, 2))
    labels: np.ndarray = random_generator.choice([0, 1], feature_quantity)
    metrics: list[qnn.Metrics] = qnn.cross_validate(features, labels)
    assert len(metrics) == fold_quantity
