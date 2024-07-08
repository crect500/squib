from __future__ import annotations

from math import ceil, log2, sqrt
from typing import Dict, List

import numpy as np
import pytest
from dask.distributed import Client, LocalCluster
from hypothesis import given
from hypothesis.strategies import integers
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

from squib.acceleration.device_setup import DaskConfig
from squib.quclidean import quclidean


@pytest.mark.parametrize(
    "index,state",
    [
        (0, [sqrt(1 / 2), sqrt(1 / 2)]),
        (1, [sqrt(1 / 2), sqrt(1 / 2)]),
        (1, [1, 0]),
        (1, [1 / 2, 1 / 2, 1 / 2, 1 / 2]),
        (3, [sqrt(1 / 2), sqrt(1 / 2)]),
        (6, [1 / 2, 1 / 2, 1 / 2, 1 / 2]),
    ],
)
def test_apply_state_to_index(index: int, state: list):
    backend: AerSimulator = AerSimulator()
    shots: int = 16384
    q: int = int(log2(len(state)))
    if index == 0:
        n: int = 1
    else:
        n: int = int(log2(index)) + 1
    qr1: QuantumRegister = QuantumRegister(1, "a")
    qr2: QuantumRegister = QuantumRegister(n, "addr")
    qr3: QuantumRegister = QuantumRegister(q, "vec")
    cr: ClassicalRegister = ClassicalRegister(q + n + 1, "p")
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3, cr)
    qc.h(qr1)
    qc.h(qr2)
    qc.append(quclidean.apply_state_to_index(index, n, 1, state), [*qr1, *qr2, *qr3])
    qc.measure([*qr1, *qr2, *qr3], cr)
    test_transpiled_circuit: QuantumCircuit = transpile(qc, backend=backend)
    result: dict = (
        backend.run(test_transpiled_circuit, shots=shots).result().get_counts()
    )
    for count, element in enumerate(state):
        address_string: str = f"{index:b}"
        address_string = ("0" * (n - len(address_string))) + address_string
        index_string: str = f"{count:b}"
        index_string = ("0" * (q - len(index_string))) + index_string
        index_string += address_string + "1"
        try:
            error: float = (
                element - sqrt((2 ** (n + 1) * result[index_string]) / shots)
            ) / element
            assert abs(error) < 10**-1
        except KeyError:
            assert element == 0


@pytest.mark.parametrize(
    "vecset",
    [
        np.asarray([[sqrt(1 / 2), sqrt(1 / 2)]]),
        np.asarray([[sqrt(1 / 2), sqrt(1 / 2)], [sqrt(1 / 2), sqrt(1 / 2)]]),
        np.asarray(
            [
                [sqrt(1 / 2), sqrt(1 / 2), 0, 0],
                [0, sqrt(1 / 2), 0, sqrt(1 / 2)],
                [sqrt(1 / 2), 0, 0, sqrt(1 / 2)],
            ],
        ),
    ],
)
def test_encode_vectors(vecset: List[list]):
    backend: AerSimulator = AerSimulator()
    shots: int = 16384
    n: int = ceil(log2(len(vecset)))
    if n == 0:
        n = 1
    q: int = int(log2(len(vecset[0])))
    qr1: QuantumRegister = QuantumRegister(1)
    qr2: QuantumRegister = QuantumRegister(n)
    qr3: QuantumRegister = QuantumRegister(q)
    cr: ClassicalRegister = ClassicalRegister(n + q + 1)
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3, cr)
    qc.h(qr1)
    qc.h(qr2)
    qc.append(quclidean.encode_vectors(vecset, n, 1), [*qr1, *qr2, *qr3])
    qc.measure([*qr1, *qr2, *qr3], cr)
    test_transpiled_circuit: QuantumCircuit = transpile(qc, backend=backend)
    result: dict = (
        backend.run(test_transpiled_circuit, shots=shots).result().get_counts()
    )
    for i, vec in enumerate(vecset):
        for j, element in enumerate(vec):
            address_string: str = f"{i:b}"
            address_string = ("0" * (n - len(address_string))) + address_string
            index_string: str = f"{j:b}"
            index_string = ("0" * (q - len(index_string))) + index_string
            index_string += address_string + "1"
            try:
                error: float = (
                    element - sqrt((2 ** (n + 1) * result[index_string]) / shots)
                ) / element
                assert abs(error) < 10**-1
            except KeyError:
                assert element == 0


@pytest.mark.parametrize(
    "vecset1,vecset2",
    [
        ([[sqrt(1 / 2), sqrt(1 / 2)]], [[sqrt(1 / 2), sqrt(1 / 2)]]),  # Test case 1
        ([[0, 1]], [[1, 0]]),
        (
            [[sqrt(1 / 2), sqrt(1 / 2)], [1, 0]],  # Test case 2
            [[sqrt(1 / 2), sqrt(1 / 2)]],
        ),
    ],
)
def test_multi_unit_euclidean(vecset1: List[list], vecset2: List[list]):
    backend: AerSimulator = AerSimulator()
    shots: int = 65536
    m: int = ceil(log2(len(vecset1)))
    if m == 0:
        m = 1
    n: int = ceil(log2(len(vecset2)))
    if n == 0:
        n = 1
    q: int = int(log2(len(vecset1[0])))
    qr1: QuantumRegister = QuantumRegister(1, "a")
    qr2: QuantumRegister = QuantumRegister(m, "i")
    qr3: QuantumRegister = QuantumRegister(n, "j")
    qr4: QuantumRegister = QuantumRegister(q, "vec")
    cr: ClassicalRegister = ClassicalRegister(m + n + q + 1)
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3, qr4, cr)
    qc.append(
        quclidean.multi_unit_euclidean(vecset1, vecset2),
        [*qr1, *qr2, *qr3, *qr4],
    )
    qc.measure([*qr1, *qr2, *qr3, *qr4], cr)
    test_transpiled_circuit: QuantumCircuit = transpile(qc, backend=backend)
    result: dict = (
        backend.run(test_transpiled_circuit, shots=shots).result().get_counts()
    )
    for i, vec1 in enumerate(vecset1):
        for j, vec2 in enumerate(vecset2):
            solution_vec: np.ndarray = (np.asarray(vec1) - np.asarray(vec2)) ** 2
            for k, solution in enumerate(solution_vec):
                index_string: str = f"{k:b}"
                index_string = ("0" * (q - len(index_string))) + index_string
                partial_index_string: str = f"{j:b}"
                partial_index_string = (
                    "0" * (m - len(index_string))
                ) + partial_index_string
                index_string += partial_index_string
                partial_index_string: str = f"{i:b}"
                partial_index_string = (
                    "0" * (n - len(index_string))
                ) + partial_index_string
                index_string += partial_index_string + "1"
                try:
                    test_result = 2 ** (m + n + 2) * result[index_string] / shots
                    error: float = (solution - test_result) / solution
                    assert abs(error) < 10**-1
                except KeyError:
                    assert solution == 0


@given(integers(min_value=1, max_value=10), integers(min_value=1, max_value=15))
def test_append_normalizer(cardinality: int, vector_size: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    vecset: np.ndarray = random_generator.normal(size=(cardinality, vector_size))
    norms: np.ndarray = np.apply_along_axis(np.linalg.norm, axis=1, arr=vecset)
    max_norm: float = np.max(norms)
    new_vector_size: int = 2 ** ceil(log2(vecset.shape[1] + 1))
    new_vecset: np.ndarray = quclidean.append_normalizers(
        vecset,
        new_vector_size,
        max_norm**2,
    )
    assert len(vecset) == len(new_vecset)
    assert np.apply_along_axis(np.linalg.norm, axis=1, arr=new_vecset) == pytest.approx(
        np.ones(cardinality),
    )
    assert new_vecset[:, 0 : vecset.shape[1]] * max_norm == pytest.approx(vecset)


@given(
    integers(min_value=1, max_value=10),
    integers(min_value=1, max_value=10),
    integers(min_value=1, max_value=15),
)
def test_build_unit_vectors(
    vecset1_cardinality: int,
    vecset2_cardinality: int,
    vector_size: int,
):
    random_generator: np.random.Generator = np.random.default_rng()
    vecset1: np.ndarray = random_generator.normal(
        size=(vecset1_cardinality, vector_size),
    )
    vecset2: np.ndarray = random_generator.normal(
        size=(vecset2_cardinality, vector_size),
    )
    new_vecset1, new_vecset2, norm = quclidean.build_unit_vectors(vecset1, vecset2)
    assert np.apply_along_axis(
        np.linalg.norm,
        axis=1,
        arr=new_vecset1,
    ) == pytest.approx(np.ones(vecset1_cardinality))
    assert np.apply_along_axis(
        np.linalg.norm,
        axis=1,
        arr=new_vecset2,
    ) == pytest.approx(np.ones(vecset2_cardinality))
    assert new_vecset1[:, 0 : vecset1.shape[1]] * norm == pytest.approx(vecset1)
    assert new_vecset2[:, 0 : vecset2.shape[1]] * norm == pytest.approx(vecset2)


@pytest.mark.parametrize(
    "set1_size,set2_size,results_dict,solution",
    [
        (1, 1, {"00001": 1, "01001": 2}, [[3]]),
        (2, 2, {"00101": 1, "10101": 3}, [[0, 4], [0, 0]]),
        (2, 2, {"00001": 2, "00101": 1, "00111": 3}, [[2, 1], [0, 3]]),
        (3, 1, {"000001": 1}, [[1], [0], [0]]),
        (
            3,
            4,
            {"0000001": 1, "0001011": 2, "0011011": 3, "0011101": 4},
            [[1, 0, 0, 0], [0, 2, 0, 3], [0, 0, 0, 4]],
        ),
    ],
)
def test_retrieve_vectors(
    set1_size: int,
    set2_size: int,
    results_dict: Dict[str, int],
    solution: List[List[float]],
):
    test_solution: np.ndarray = quclidean.retrieve_vectors(
        set1_size,
        set2_size,
        results_dict,
    )
    assert test_solution.shape == (set1_size, set2_size)
    for i, vector in enumerate(solution):
        for j, element in enumerate(vector):
            assert element == test_solution[i][j]


@pytest.mark.parametrize(
    "vecset1,vecset2",
    [
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]),
        ([[1.0, 2.0, 3.0]], [[3.0, 4.0, 5.0]]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0, 3.0]]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ),
    ],
)
def test_multi_euclidean4d(vecset1: List[List[float]], vecset2: List[List[float]]):
    vecset1: np.ndarray = np.asarray(vecset1)
    vecset2: np.ndarray = np.asarray(vecset2)
    test_solution: np.ndarray = quclidean.multi_euclidean(
        vecset1,
        vecset2,
        shots=65536,
    )
    for i, vec1 in enumerate(vecset1):
        for j, vec2 in enumerate(vecset2):
            difference: np.ndarray = vec1 - vec2
            solution: float = float(np.dot(difference, difference))
            if solution == 0:
                assert test_solution[i][j] == 0
            else:
                error: float = (solution - test_solution[i][j]) / solution
                assert abs(error) < 10**-1


@pytest.mark.parametrize(
    ("vectors", "jobs", "first_vector_size", "last_vector_size"),
    [
        (np.ndarray((1, 1)), 1, 1, 1),
        (np.ndarray((2, 1)), 1, 2, 2),
        (np.ndarray((2, 1)), 2, 1, 1),
        (np.ndarray((3, 1)), 2, 2, 1),
        (np.ndarray((5, 1)), 2, 3, 2),
        (np.ndarray((31, 3)), 15, 3, 2),
    ],
)
def test_create_partitions(
    vectors: np.ndarray,
    jobs: int,
    first_vector_size: int,
    last_vector_size: int,
) -> None:
    partitions: list[np.ndarray] = quclidean.create_partitions(vectors, jobs)
    assert len(partitions) == jobs
    assert len(partitions[0]) == first_vector_size
    assert len(partitions[-1]) == last_vector_size


@pytest.mark.parametrize(
    "vecset1,vecset2",
    [
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]),
        ([[1.0, 2.0, 3.0]], [[3.0, 4.0, 5.0]]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0, 3.0]]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ),
    ],
)
def test_multi_euclidean4d_from_gate(
    vecset1: list[list[float]],
    vecset2: list[list[float]],
) -> None:
    vecset1_array: np.ndarray = np.asarray(vecset1)
    vecset2_array: np.ndarray = np.asarray(vecset2)
    normalized_vecset1, normalized_vecset2, norm = quclidean.build_unit_vectors(
        vecset1_array,
        vecset2_array,
    )
    vecset2_circuit: QuantumCircuit = quclidean.create_vecset_gate(
        normalized_vecset2,
        as_circuit=True,
    )
    test_solution: np.ndarray = quclidean.multi_euclidean_from_gate(
        normalized_vecset1,
        vecset2_circuit,
        len(normalized_vecset2),
        norm,
        shots=2**20,
    )
    for i, vec1 in enumerate(vecset1_array):
        for j, vec2 in enumerate(vecset2_array):
            difference: np.ndarray = vec1 - vec2
            solution: float = float(np.dot(difference, difference))
            if solution == 0:
                assert test_solution[i][j] == 0
            else:
                error: float = (solution - test_solution[i][j]) / solution
                assert abs(error) < 10**-1


@pytest.mark.parametrize(
    ("vecset1", "vecset2", "available_processors"),
    [
        ([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]], 1),
        ([[1.0, 2.0, 3.0]], [[3.0, 4.0, 5.0]], 1),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0, 3.0]], 2),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 2),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            2,
        ),
        (
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            2,
        ),
    ],
)
def test_multi_circuit_multi_euclidean4d(
    vecset1: List[List[float]],
    vecset2: List[List[float]],
    available_processors: int,
):
    vecset1: np.ndarray = np.asarray(vecset1)
    vecset2: np.ndarray = np.asarray(vecset2)
    device_config: DaskConfig = DaskConfig("fake_filepath")
    device_config.mode = "local"
    device_config.jobs = available_processors
    device_config.client = Client(LocalCluster(available_processors))
    test_solution: np.ndarray = quclidean.multi_circuit_multi_euclidean(
        vecset1,
        vecset2,
        shots=65536,
        device_config=device_config,
    )
    for i, vec1 in enumerate(vecset1):
        for j, vec2 in enumerate(vecset2):
            difference: np.ndarray = vec1 - vec2
            solution: float = float(np.dot(difference, difference))
            if solution == 0:
                assert test_solution[i][j] == 0
            else:
                error: float = (solution - test_solution[i][j]) / solution
                assert abs(error) < 10**-1


@pytest.mark.parametrize(
    ("list1, list2, quant_results, solution"),
    [
        ([[0, 0, 0]], [[0, 0, 0]], {"00001": 0, "01001": 0, "10001": 0}, 0),
        ([[1, 0, 0]], [[1, 0, 0]], {"00001": 0, "01001": 0, "10001": 0}, 0),
        ([[1, 0, 1]], [[1, 0, 1]], {"00001": 0, "01001": 0, "10001": 0}, 0),
        ([[0, 0, 0]], [[0, 0, 0]], {"00001": 1, "01001": 0, "10001": 0}, 1),
        ([[1, 0, 0]], [[0, 0, 0]], {"00001": 1, "01001": 0, "10001": 0}, 0),
        (
            [[0, 0, 0], [0, 0, 1]],
            [[0, 0, 0]],
            {"00001": 0, "01001": 0, "10001": 0, "00011": 0, "01011": 0, "10011": 1},
            0,
        ),
        (
            [[0, 0, 0], [0, 0, 1]],
            [[0, 0, 0]],
            {"00001": 0, "01001": 0, "10001": 0, "00011": 0, "01011": 0, "10011": 0},
            0.5,
        ),
    ],
)
def test_find_error_results(
    list1: List[np.ndarray],
    list2: List[np.ndarray],
    quant_results: Dict[str, int],
    solution: float,
):
    list1: np.ndarray = np.asarray(list1)
    list2: np.ndarray = np.asarray(list2)
    distances: np.ndarray = quclidean.retrieve_vectors(
        len(list1),
        len(list2),
        quant_results,
    )
    assert quclidean.compare_euclideans(list1, list2, distances) == solution
