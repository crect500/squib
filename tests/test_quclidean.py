from __future__ import annotations

from math import ceil, log2, sqrt

import numpy as np
import pytest
from dask.distributed import Client, LocalCluster
from hypothesis import given, settings
from hypothesis.strategies import integers
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from squib.acceleration.device_setup import DaskConfig
from squib.quclidean import quclidean

SHOTS13: int = 8192
SHOTS14: int = 16384
SHOTS16: int = 65536
SHOTS18: int = 262144


@given(integers(min_value=0, max_value=8), integers(min_value=1, max_value=3))
@settings(deadline=500, max_examples=5)
def test_apply_state_to_index(index: int, vector_qubit_quantity: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    state: np.ndarray = random_generator.normal(size=2**vector_qubit_quantity)
    state = state / np.linalg.norm(state)
    backend: AerSimulator = AerSimulator()
    shots: int = SHOTS13
    vector_qubit_quantity: int = int(log2(len(state)))
    if index == 0:
        n: int = 1
    else:
        n: int = int(log2(index)) + 1
    qr1: QuantumRegister = QuantumRegister(1, "a")
    qr2: QuantumRegister = QuantumRegister(n, "j")
    qr3: QuantumRegister = QuantumRegister(vector_qubit_quantity, "vec")
    cr: ClassicalRegister = ClassicalRegister(vector_qubit_quantity + n + 1, "p")
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3, cr)
    qc.h(qr1)
    qc.h(qr2)
    qc.compose(
        quclidean.apply_state_to_index(index, n, 1, state, as_circuit=True),
        [*qr1, *qr2, *qr3],
        inplace=True,
    )
    qc.measure([*qr1, *qr2, *qr3], cr)
    result: dict = backend.run(qc, shots=shots).result().get_counts()
    for count, element in enumerate(state):
        address_string: str = f"{index:0{n}b}"
        index_string: str = f"{count:0{vector_qubit_quantity}b}"
        index_string += address_string + "1"
        retrieved_value: int = result.get(index_string)
        if retrieved_value is not None:
            assert sqrt(2 ** (n + 1) * retrieved_value / shots) == pytest.approx(
                abs(element),
                rel=1e-2,
                abs=1e-1,
            )
        else:
            assert element == pytest.approx(0, abs=1e-1)


@given(integers(min_value=1, max_value=8), integers(min_value=1, max_value=3))
@settings(deadline=5000, max_examples=5)
def test_encode_vectors(vector_quantity: int, vector_qubits: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    vecset: np.ndarray = random_generator.normal(
        size=(vector_quantity, 2**vector_qubits),
    )
    for index, vector in enumerate(vecset):
        vecset[index] = vector / np.linalg.norm(vector)
    backend: AerSimulator = AerSimulator()
    shots: int = SHOTS13
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
    qc.compose(
        quclidean.encode_vectors(vecset, n, 1, as_circuit=True),
        [*qr1, *qr2, *qr3],
        inplace=True,
    )
    qc.measure([*qr1, *qr2, *qr3], cr)
    result: dict = backend.run(qc, shots=shots).result().get_counts()
    for i, vec in enumerate(vecset):
        for j, element in enumerate(vec):
            address_string: str = f"{i:0{n}b}"
            index_string: str = f"{j:0{q}b}"
            index_string += address_string + "1"
            retrieved_value: int = result.get(index_string)
            if retrieved_value is not None:
                assert sqrt(
                    2 ** (n + 1) * retrieved_value / shots,
                ) == pytest.approx(abs(element), rel=1e-2, abs=1e-1)
            else:
                assert element == pytest.approx(0, abs=1e-1)


@given(
    integers(min_value=1, max_value=2),
    integers(min_value=1, max_value=4),
    integers(min_value=1, max_value=3),
)
@settings(deadline=10000, max_examples=5)
def test_multi_unit_euclidean(
    vecset1_size: int,
    vecset2_size: int,
    vector_qubits: int,
) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    vecset1: np.ndarray = random_generator.normal(size=(vecset1_size, 2**vector_qubits))
    vecset2: np.ndarray = random_generator.normal(size=(vecset2_size, 2**vector_qubits))
    for index, vector in enumerate(vecset1):
        vecset1[index] = vector / np.linalg.norm(vector)
    for index, vector in enumerate(vecset2):
        vecset2[index] = vector / np.linalg.norm(vector)
    backend: AerSimulator = AerSimulator()
    shots: int = SHOTS16
    m: int = ceil(log2(vecset1_size))
    if m == 0:
        m = 1
    n: int = ceil(log2(vecset2_size))
    if n == 0:
        n = 1
    qr1: QuantumRegister = QuantumRegister(1, "a")
    qr2: QuantumRegister = QuantumRegister(m, "i")
    qr3: QuantumRegister = QuantumRegister(n, "j")
    qr4: QuantumRegister = QuantumRegister(vector_qubits, "vec")
    cr: ClassicalRegister = ClassicalRegister(m + n + vector_qubits + 1)
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3, qr4, cr)
    qc.compose(
        quclidean.multi_unit_euclidean(vecset1, vecset2, as_circuit=True),
        [*qr1, *qr2, *qr3, *qr4],
        inplace=True,
    )
    qc.measure([*qr1, *qr2, *qr3, *qr4], cr)
    result: dict = backend.run(qc, shots=shots).result().get_counts()
    for i, vec1 in enumerate(vecset1):
        for j, vec2 in enumerate(vecset2):
            solution_vec: np.ndarray = (np.asarray(vec1) - np.asarray(vec2)) ** 2
            for k, solution in enumerate(solution_vec):
                index_string: str = f"{k:0{vector_qubits}b}"
                index_string = (
                    "0" * (vector_qubits - len(index_string))
                ) + index_string
                partial_index_string: str = f"{j:0{n}b}"
                index_string += partial_index_string
                partial_index_string: str = f"{i:0{m}b}"
                index_string += partial_index_string + "1"
                retrieved_value: int = result.get(index_string)
                if retrieved_value is not None:
                    value: float = (2 ** (m + n + 2) * retrieved_value) / shots
                    assert value == pytest.approx(solution, rel=1e-2, abs=1e-1)
                else:
                    assert solution == pytest.approx(0, abs=1e-1)


@given(integers(min_value=1, max_value=10), integers(min_value=1, max_value=15))
def test_append_normalizer(cardinality: int, vector_size: int) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    vecset: np.ndarray = random_generator.normal(size=(cardinality, vector_size))
    norms: np.ndarray = np.apply_along_axis(np.linalg.norm, axis=1, arr=vecset)
    max_norm: float = np.max(norms)
    new_vector_size: int = 2 ** ceil(log2(vecset.shape[1] + 1))
    new_vecset: np.ndarray = quclidean._append_normalizers(
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
) -> None:
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
    ("set1_size", "set2_size", "results_dict", "solution"),
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
    results_dict: dict[str, int],
    solution: list[list[float]],
) -> None:
    test_solution: np.ndarray = quclidean.retrieve_vectors(
        set1_size,
        set2_size,
        3,
        2,
        results_dict,
    )
    assert test_solution.shape == (set1_size, set2_size)
    for i, vector in enumerate(solution):
        for j, element in enumerate(vector):
            assert element == test_solution[i][j]


@given(
    integers(min_value=1, max_value=5),
    integers(min_value=1, max_value=5),
    integers(min_value=1, max_value=3),
)
@settings(deadline=15000)
def test_multi_euclidean(
    vecset1_size: int,
    vecset2_size: int,
    vector_qubits: int,
) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    vecset1: np.ndarray = random_generator.normal(size=(vecset1_size, 2**vector_qubits))
    vecset2: np.ndarray = random_generator.normal(size=(vecset2_size, 2**vector_qubits))
    test_solution: np.ndarray = quclidean.multi_euclidean(
        vecset1,
        vecset2,
        shots=SHOTS18,
    )
    for i, vec1 in enumerate(vecset1):
        for j, vec2 in enumerate(vecset2):
            difference: np.ndarray = vec1 - vec2
            solution: float = float(np.dot(difference, difference))
            if solution == 0:
                assert test_solution[i][j] == pytest.approx(0, abs=1e-1)
            else:
                assert test_solution[i][j] == pytest.approx(
                    solution,
                    rel=2e-1,
                    abs=2e-1,
                )


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
    partitions: list[np.ndarray] = quclidean._create_partitions(vectors, jobs)
    assert len(partitions) == jobs
    assert len(partitions[0]) == first_vector_size
    assert len(partitions[-1]) == last_vector_size


@given(
    integers(min_value=1, max_value=5),
    integers(min_value=1, max_value=5),
    integers(min_value=1, max_value=3),
)
@settings(deadline=15000)
def test_multi_euclidean_from_gate(
    vecset1_size: int,
    vecset2_size: int,
    vector_qubits: int,
) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    vecset1: np.ndarray = random_generator.normal(size=(vecset1_size, 2**vector_qubits))
    vecset2: np.ndarray = random_generator.normal(size=(vecset2_size, 2**vector_qubits))
    vector_size: int = vecset1.shape[1]
    normalized_vecset1, normalized_vecset2, norm = quclidean.build_unit_vectors(
        vecset1,
        vecset2,
    )
    vecset2_circuit: QuantumCircuit = quclidean._create_vecset_gate(
        normalized_vecset2,
        as_circuit=True,
    )
    test_solution: np.ndarray = quclidean.multi_euclidean_from_gate(
        normalized_vecset1,
        vecset2_circuit,
        len(normalized_vecset2),
        vector_size,
        norm,
        shots=SHOTS18,
    )
    for i, vec1 in enumerate(vecset1):
        for j, vec2 in enumerate(vecset2):
            difference: np.ndarray = vec1 - vec2
            solution: float = float(np.dot(difference, difference))
            if solution == 0:
                assert test_solution[i][j] == pytest.approx(0, rel=2e-1, abs=2e-1)
            else:
                assert test_solution[i][j] == pytest.approx(
                    solution,
                    rel=2e-1,
                    abs=2e-1,
                )


@given(
    integers(min_value=1, max_value=5),
    integers(min_value=1, max_value=5),
    integers(min_value=1, max_value=3),
    integers(min_value=1, max_value=4),
)
@settings(deadline=2000)
def test_multi_circuit_multi_euclidean(
    vecset1_size: int,
    vecset2_size: int,
    vector_qubits: int,
    available_processors: int,
) -> None:
    random_generator: np.random.Generator = np.random.default_rng()
    vecset1: np.ndarray = random_generator.normal(size=(vecset1_size, 2**vector_qubits))
    vecset2: np.ndarray = random_generator.normal(size=(vecset2_size, 2**vector_qubits))
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
    ("list1", "list2", "quant_results", "solution"),
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
    list1: list[np.ndarray],
    list2: list[np.ndarray],
    quant_results: dict[str, int],
    solution: float,
) -> None:
    list1: np.ndarray = np.asarray(list1)
    list2: np.ndarray = np.asarray(list2)
    vector_qubits: int = ceil(log2(list1.shape[1] + 1))
    distances: np.ndarray = quclidean.retrieve_vectors(
        len(list1),
        len(list2),
        list1.shape[1],
        vector_qubits,
        quant_results,
    )
    assert quclidean.compare_euclideans(list1, list2, distances) == solution
