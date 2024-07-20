"""Implements custom quantum circuits applicable to multiple projects."""

from __future__ import annotations

import logging
from math import ceil, log2, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

if TYPE_CHECKING:
    from qiskit.circuit import Gate

    from squib.acceleration.device_setup import DaskConfig

logger: logging.Logger = logging.getLogger(__name__)
QISKIT_PARALLEL: bool = True


def apply_state_to_index(  # noqa: PLR0913
    index: int,
    register_size: int,
    control_state: int,
    vec: np.ndarray,
    backend: AerSimulator | None = None,
    as_circuit: bool = False,  # noqa: FBT001, FBT002
) -> Gate | QuantumCircuit:
    """
    Apply an amplitude encoding to the indicated state.

    Args:
    ----
    index: Index to apply the state preparation to.
    register_size: The size of the quantum address register.
    control_state: The ancillary qubit state on which to apply encoding
    vec: The vector which to amplitude encode into the quantum register.
    backend: Optionally specify a backend other than the AerSimulator.
    as_circuit: Returns a circuit if true, a gate if false. Default is false.

    Returns:
    -------
    The resulting gate or circuit

    Raises:
    ------
    ValueError if the input vector is not a power of 2

    """
    if not backend:
        backend = AerSimulator()
    q: float = log2(len(vec))
    if not q.is_integer() or q == 0:
        raise ValueError(
            "The input vector size must be a power of 2. "
            "They are of length " + str(len(vec)),
        )
    q: int = int(q)
    if control_state == 0:
        address_register_name: str = "i"
    else:
        address_register_name = "j"
    qr1: QuantumRegister = QuantumRegister(1, "a")
    qr2: QuantumRegister = QuantumRegister(register_size, address_register_name)
    qr3: QuantumRegister = QuantumRegister(q, "vec")
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3)
    index = (index << 1) + 1 if control_state else index << 1
    state_gate = StatePreparation(vec, label="data")
    controlled_gate = state_gate.control(register_size + 1, ctrl_state=index)
    qc.append(controlled_gate, [*qr1, *qr2, *qr3])

    if as_circuit:
        return transpile(qc, backend=backend, optimization_level=3)

    return transpile(qc, backend=backend, optimization_level=3).to_gate(label="index")


def encode_vectors(  # noqa: PLR0913
    vecset: np.ndarray,
    register_size: int,
    control_state: int,
    *,
    backend: AerSimulator | None = None,
    as_circuit: bool = False,
    device_config: DaskConfig | None = None,
) -> Gate | QuantumCircuit:
    """
    Encode a set of vectors into their corresponding indices of a quantum state.

    Args:
    ----
    vecset: A set of normalized vectors.
    register_size: The size of the quantum address register.
    control_state: Whether to apply the gate to a 0 or 1 state ancillary qubit.
    backend: The backend on which to execute the circuit
    as_circuit: Returns a circuit if true, a gate if false. Default is false.
    device_config: The configuration of the executor

    Returns:
    -------
    The resulting gate or circuit

    """
    if not backend:
        backend = AerSimulator()
    n: int = ceil(log2(len(vecset[0])))
    if n == 0:
        n = 1
    qr1: QuantumRegister = QuantumRegister(1, "a")
    if control_state == 0:
        address_register_name: str = "i"
    else:
        address_register_name = "j"
    qr2: QuantumRegister = QuantumRegister(register_size, address_register_name)
    qr3: QuantumRegister = QuantumRegister(n, "vec")
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3)
    if device_config:
        logger.warning("Scattering backends...")
        backend_scattered = device_config.client.scatter([backend] * len(vecset))
        logger.warning("Scattering complete.")
        futures = device_config.client.map(
            apply_state_to_index,
            list(range(len(vecset))),
            [register_size] * len(vecset),
            [control_state] * len(vecset),
            vecset,
            backend_scattered,
            [True] * len(vecset),
        )
        logger.warning("Gathering circuits...")
        circuit_list: list[Gate] = device_config.client.gather(futures)
        futures.clear()
        for i, vector_circuit in enumerate(circuit_list):
            logging.warning("Appending circuit %s / %s\r", str(i), str(len(vecset) - 1))
            qc.compose(vector_circuit, [*qr1, *qr2, *qr3], inplace=True)
    else:
        for i, vec in enumerate(vecset):
            logging.warning("Appending circuit %s / %s\r", str(i), str(len(vecset) - 1))
            if np.sum(vec) != 0:
                qc.compose(
                    apply_state_to_index(
                        i,
                        register_size,
                        control_state,
                        vec,
                        backend=backend,
                        as_circuit=True,
                    ),
                    [*qr1, *qr2, *qr3],
                    inplace=True,
                )

    logger.warning("Returning circuit...")
    if as_circuit:
        return qc

    return qc.to_gate(label="encode")


def _check_vector_sizes(vecset1: np.ndarray, vecset2: np.ndarray) -> int:
    """
    Raise ValueError if any of the vectors are different sizes.

    Args:
    ----
    vecset1: 2D array with vectors as rows
    vecset2: 2D array with vectors as rows

    Returns:
    -------
    The length of the vectors

    Raises:
    ------
    ValueError if not all vectors are the same length

    """
    if len(vecset1) == 0:
        raise ValueError("The vectors must be nonempty")  # noqa: TRY003
    length: int = len(vecset1[0])
    for count, vector in enumerate(vecset1):
        if len(vector) != length:
            raise ValueError(
                "Index " + str(count) + " of vector 1 has a mismatched size",
            )
    for count, vector in enumerate(vecset2):
        if len(vector) != length:
            raise ValueError(
                "Index " + str(count) + " of vector 2 has a mismatched size",
            )

    return length


def multi_unit_euclidean(
    vecset1: np.ndarray,
    vecset2: np.ndarray,
    *,
    backend: AerSimulator | None = None,
    as_circuit: bool = False,
    device_config: DaskConfig | None = None,
) -> Gate | QuantumCircuit:
    """
    Create quantum circuit which finds the Euclidean distance between all vectors.

    Args:
    ----
    vecset1: 2D array with vectors as rows
    vecset2: 2D array with vectors as rows
    backend: The backend on which to execute the circuit
    as_circuit: Returns a circuit if true, a gate if false. Default is false.
    device_config: The configuration of the executor

    Returns:
    -------
    The resulting gate or circuit

    Raises:
    ------
    ValueError if the vectors are empty or vectors are not of uniform size

    """
    if not backend:
        backend = AerSimulator()

    length: int = _check_vector_sizes(vecset1, vecset2)

    m: int = ceil(log2(len(vecset1)))
    if m == 0:
        m = 1
    n: int = ceil(log2(len(vecset2)))
    if n == 0:
        n = 1
    q: float = log2(length)
    if not q.is_integer() or q == 0:
        raise ValueError(
            "The input vector size must be a power of 2."
            " They are of length " + str(length),
        )
    q: int = int(q)
    qr1: QuantumRegister = QuantumRegister(1, "a")
    qr2: QuantumRegister = QuantumRegister(m, "i")
    qr3: QuantumRegister = QuantumRegister(n, "j")
    qr4: QuantumRegister = QuantumRegister(q, "vec")
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3, qr4)
    qc.h(qr1)
    qc.h(qr2)
    qc.h(qr3)
    vector_circuit: Gate = encode_vectors(
        vecset1,
        m,
        0,
        backend=backend,
        device_config=device_config,
        as_circuit=True,
    )
    qc.compose(vector_circuit, [*qr1, *qr2, *qr4], inplace=True)
    vector_circuit: Gate = encode_vectors(
        vecset2,
        n,
        1,
        backend=backend,
        device_config=device_config,
        as_circuit=True,
    )
    qc.compose(vector_circuit, [*qr1, *qr3, *qr4], inplace=True)
    qc.h(qr1)

    if as_circuit:
        return qc

    return qc.to_gate()


def _append_normalizers(
    vecset: np.ndarray,
    new_vector_size: int,
    norm_squared: float,
) -> np.ndarray:
    """
    Reshape and append normalizer to original vector.

    Args:
    ----
    vecset: A real-valued 1D vector
    new_vector_size: The size of the vector to create
    norm_squared: The norm of the new vector to be created

    Returns:
    -------
    The new vector with specified norm and size

    Raises:
    ------
    ValueError if calculation produces a negative input to square root function

    """
    new_vecset: np.ndarray = np.zeros(
        (vecset.shape[0], new_vector_size),
        dtype=float,
    )
    new_vecset[:, 0 : vecset.shape[1]] = vecset
    norm: float = sqrt(norm_squared)
    if norm_squared == 0:
        return new_vecset

    for index, vectors in enumerate(zip(vecset, new_vecset)):
        old_vector: np.ndarray = vectors[0]
        new_vector: np.ndarray = vectors[1]
        normalizer_squared: float = norm_squared - np.dot(old_vector, old_vector)
        if normalizer_squared <= 0:
            if norm_squared - np.dot(old_vector, old_vector) > -(2**-5):
                new_vecset[index] = new_vector * (1 / norm)
                continue
            raise ValueError(
                "Math domain error for norm_squared "
                + str(norm_squared)
                + " and dot product "
                + str(np.dot(old_vector, old_vector)),
            ) from None

        new_vector[vecset.shape[1]] = sqrt(normalizer_squared)
        new_vecset[index] = new_vector * (1 / norm)

    return new_vecset


def build_unit_vectors(
    vecset1: np.ndarray,
    vecset2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert vectors to unit vectors by appending an additional scaling element.

    Args:
    ----
    vecset1: 2D array with vectors as rows
    vecset2: 2D array with vectors as rows

    Returns:
    -------
    The new vectors and the maximum norm

    Raises:
    ------
    ValueError if vectors second dimensions are not the same size

    """
    if vecset1.shape[1] != vecset2.shape[1]:
        message: str = (
            f"Size mismatch between vectors of length {vecset1.shape[1]} "
            f"and length {vecset2.shape[1]}"
        )
        logger.error(message)
        raise ValueError(message)

    norms: np.ndarray = np.apply_along_axis(np.linalg.norm, axis=1, arr=vecset1)
    norms = np.append(
        norms,
        np.apply_along_axis(np.linalg.norm, axis=1, arr=vecset2),
        axis=0,
    )
    norm: float = np.max(norms)
    norm_squared: float = norm**2
    new_vector_size: int = 2 ** ceil(log2(vecset1.shape[1] + 1))

    new_vecset1: np.ndarray = _append_normalizers(
        vecset1,
        new_vector_size,
        norm_squared,
    )
    new_vecset2: np.ndarray = _append_normalizers(
        vecset2,
        new_vector_size,
        norm_squared,
    )

    return new_vecset1, new_vecset2, norm


def retrieve_vectors(
    set1_size: int,
    set2_size: int,
    vector_size: int,
    vector_qubits: int,
    results: dict[str, int],
) -> [np.ndarray]:
    """
    Store the results of a quantum circuit dictionary into proper array elements.

    Args:
    ----
    set1_size: The number of vectors in the first set
    set2_size: The number of vectors in the second set
    vector_size: The size of the vectors within each set
    vector_qubits: The qybit quantity of the vector data register
    results: The results of a quantum circuit

    Returns:
    -------
    The resolved vectors in correct indices

    """
    m: int = ceil(log2(set1_size))
    if m == 0:
        m = 1
    n: int = ceil(log2(set2_size))
    if n == 0:
        n = 1

    distance_matrix: np.ndarray = np.zeros((set1_size, set2_size), dtype=np.float64)
    for i in range(set1_size):
        for j in range(set2_size):
            index_string: str = f"{j:0{n}b}"
            partial_index_string: str = f"{i:0{m}b}"
            index_string += partial_index_string + "1"
            total: float = 0
            for k in range(vector_size):
                vector_index_string = f"{k:0{vector_qubits}b}" + index_string
                retrieved_value: int = results.get(vector_index_string)
                if retrieved_value is not None:
                    total += retrieved_value
            distance_matrix[i][j] = total

    return distance_matrix


def multi_euclidean(
    vecset1: np.ndarray,
    vecset2: np.ndarray,
    backend: Any | None = None,  # noqa: ANN401
    shots: int = 16384,
    *,
    device_config: DaskConfig | None = None,
) -> np.ndarray:
    """
    Calculate the Euclidean distance between each point in two sets of vectors.

    Args:
    ----
    vecset1: A list of three-dimensional vectors.
    vecset2: A list of three-dimensional vectors.
    backend: Backend on which to run the quantum circuit. Default is
    qiskit_aer.AerSimulator
    shots: The number of times to execute the quantum circuit.
    device_config: The configuration of the executor

    Returns:
    -------
    A matrix of distances from each vector of set1 to that of set1. The row index
    indices the first set index. The column indicates the second set index.

    """
    if not backend:
        backend = AerSimulator()
    if vecset1.ndim == 1:
        vecset1.resize((1, vecset1.shape[0]))
    vector_size: int = vecset1.shape[1]
    vecset1, vecset2, norm = build_unit_vectors(vecset1, vecset2)
    m: int = ceil(log2(len(vecset1)))
    if m == 0:
        m = 1
    n: int = ceil(log2(len(vecset2)))
    if n == 0:
        n = 1
    vector_qubits: int = ceil(log2(vector_size + 1))
    qr1: QuantumRegister = QuantumRegister(1, "a")
    qr2: QuantumRegister = QuantumRegister(m, "i")
    qr3: QuantumRegister = QuantumRegister(n, "j")
    qr4: QuantumRegister = QuantumRegister(vector_qubits, "vec")
    cr: ClassicalRegister = ClassicalRegister(m + n + vector_qubits + 1)
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3, qr4, cr)
    qc.compose(
        multi_unit_euclidean(
            vecset1,
            vecset2,
            device_config=device_config,
            as_circuit=True,
        ),
        [*qr1, *qr2, *qr3, *qr4],
        inplace=True,
    )
    qc.measure([*qr1, *qr2, *qr3, *qr4], cr)
    logger.warning("Transpiling")
    if device_config:
        transpiled_circuit: QuantumCircuit = transpile(
            qc,
            backend=backend,
            optimization_level=0,
        )
    else:
        transpiled_circuit = transpile(qc, backend=backend, optimization_level=0)
    logger.warning("Executing")
    if device_config:
        result: dict = (
            backend.run(
                transpiled_circuit,
                shots=shots,
                executor=device_config.client,
            )
            .result()
            .get_counts()
        )
    else:
        result: dict = (
            backend.run(transpiled_circuit, shots=shots).result().get_counts()
        )
    distances: np.ndarray = (
        retrieve_vectors(
            len(vecset1),
            len(vecset2),
            vector_size,
            vector_qubits,
            result,
        )
        * 2 ** (m + n + 2)
        * norm**2
    ) / shots

    return distances


def _create_partitions(vecset: np.ndarray, available_jobs: int) -> list[np.ndarray]:
    """
    Split vector set into partitions to be scattered as tasks.

    Args:
    ----
    vecset: A set of vectors
    available_jobs: The number of available tasks to scatter to

    Returns:
    -------
    A list of smaller vector sets

    """
    vecset_partitions: list[np.ndarray] = []
    divisor: int = int(len(vecset) / available_jobs)
    remainder: int = len(vecset) % available_jobs
    start: int = 0
    if remainder != 0:
        end: int = divisor + 1
    else:
        end: int = divisor
    for _ in range(remainder):
        vecset_partitions.append(vecset[start:end])
        start += divisor + 1
        end += divisor + 1
    if remainder != 0:
        end -= 1
    for _ in range(remainder, available_jobs):
        vecset_partitions.append(vecset[start:end])
        start += divisor
        end += divisor

    return vecset_partitions


def _create_vecset_gate(
    vecset: np.ndarray,
    backend: AerSimulator | None = None,
    device_config: DaskConfig | None = None,
    *,
    as_circuit: bool = False,
) -> Gate | QuantumCircuit:
    """
    Create a gate to encode only one vector set.

    Args:
    ----
    vecset: A set of vectors
    backend: The execution backend
    device_config: The configuration of the executor
    as_circuit: Optionally return a circuit

    Returns:
    -------
    The resulting gate or circuit

    """
    if not backend:
        backend = AerSimulator()
    n: int = ceil(log2(len(vecset)))
    if n == 0:
        n = 1
    if not device_config:
        return encode_vectors(vecset, n, control_state=1, as_circuit=as_circuit)
    return encode_vectors(
        vecset,
        n,
        control_state=1,
        backend=backend,
        device_config=device_config,
        as_circuit=as_circuit,
    )


def multi_euclidean_from_gate(  # noqa: PLR0913
    vecset1: np.ndarray,
    vecset2_circuit: QuantumCircuit,
    vecset2_size: int,
    vector_size: int,
    norm: float,
    backend: AerSimulator | None = None,
    shots: int = 16384,
) -> np.ndarray:
    """
    Perform quantum euclidean algorithm was encoding gate provided.

    Args:
    ----
    vecset1: A list of three-dimensional vectors.
    vecset2_circuit: A circuit encoding a list of three-dimensional vectors
    vecset2_size: The size of the second vector set
    vector_size: The size of the original vector
    norm: The norm of all the modified vectors
    backend: The executor backend
    shots: The number of times to measure the executed circuit

    """
    if not backend:
        backend = AerSimulator()
    if vecset1.ndim == 1:
        vecset1.resize((1, vecset1.shape[0]))
    m: int = ceil(log2(len(vecset1)))
    if m == 0:
        m = 1
    n: int = ceil(log2(vecset2_size))
    if n == 0:
        n = 1
    vector_qubits: int = ceil(log2(vector_size + 1))
    qr1: QuantumRegister = QuantumRegister(1, "a")
    qr2: QuantumRegister = QuantumRegister(m, "i")
    qr3: QuantumRegister = QuantumRegister(n, "j")
    qr4: QuantumRegister = QuantumRegister(vector_qubits, "vec")
    cr: ClassicalRegister = ClassicalRegister(m + n + vector_qubits + 1)
    qc: QuantumCircuit = QuantumCircuit(qr1, qr2, qr3, qr4, cr)
    qc.h(qr1)
    qc.h(qr2)
    qc.h(qr3)
    vecset1_circuit: QuantumCircuit = encode_vectors(
        vecset1,
        m,
        0,
        backend=backend,
        as_circuit=True,
    )
    logger.warning("Composing...")
    qc.compose(vecset1_circuit, [*qr1, *qr2, *qr4], inplace=True)
    qc.compose(vecset2_circuit, [*qr1, *qr3, *qr4], inplace=True)
    qc.h(qr1)

    qc.measure([*qr1, *qr2, *qr3, *qr4], cr)
    logger.warning("Executing")
    result: dict = backend.run(qc, shots=shots).result().get_counts()
    return (
        (
            retrieve_vectors(
                len(vecset1),
                vecset2_size,
                vector_size,
                vector_qubits,
                result,
            )
        )
        * 2 ** (m + n + 2)
        * norm**2
        / shots
    )


def distributed_multi_euclidean(  # noqa: PLR0913
    vecset1_partitions: list[np.ndarray],
    vecset2: np.ndarray,
    vector_size: int,
    device_config: DaskConfig,
    norm: float,
    backend: AerSimulator | None = None,
    *,
    shots: int = 16384,
) -> np.ndarray:
    """
    Perform Euclidean distance calculations by executing circuits in parallel.

    Args:
    ----
    vecset1_partitions: Partitioned vector sets
    vecset2: A list of vectors.
    vector_size: The size of the original vector
    device_config: The configuration of the executor
    norm: The norm of all the modified vectors
    backend: The executor backend
    shots: The number of times to measure the executed circuit

    Returns:
    -------
    The Euclidean distances as a 2D array

    """
    vecset2_circuit: Gate = _create_vecset_gate(
        vecset2,
        device_config=device_config,
        as_circuit=True,
    )
    vecset1_size: int = 0
    for vecset in vecset1_partitions:
        vecset1_size += len(vecset)
    distances: np.ndarray = np.ndarray((vecset1_size, len(vecset2)), dtype=np.float64)
    logger.warning("Scattering data...")
    vecset1_partitions_scattered = device_config.client.scatter(vecset1_partitions)
    backend_scattered = device_config.client.scatter(
        [backend] * len(vecset1_partitions),
    )
    logger.warning("Scattering complete.")
    futures = device_config.client.map(
        multi_euclidean_from_gate,
        vecset1_partitions_scattered,
        [vecset2_circuit] * len(vecset1_partitions),
        [len(vecset2)] * len(vecset1_partitions),
        [vector_size] * len(vecset1_partitions),
        [norm] * len(vecset1_partitions),
        backend_scattered,
        [shots] * len(vecset1_partitions),
    )
    logger.warning("Gathering results...")
    results: list[np.ndarray] = device_config.client.gather(futures)
    futures.clear()
    starting_index: int = 0
    for partition, result in zip(vecset1_partitions, results):
        distances[starting_index : starting_index + len(partition)] = result
        starting_index += len(partition)

    return distances


def multi_circuit_multi_euclidean(
    vecset1: np.ndarray,
    vecset2: np.ndarray,
    backend: Any | None = None,  # noqa: ANN401
    shots: int = 16384,
    *,
    device_config: DaskConfig | None = None,
) -> np.ndarray:
    """
    Run quantum euclidean distance as multiple circuits.

    Partitions the first set of vectors into single vectors or pairs of vectors to
    provide to the multi_euclidean method. This reduces transpile time and
    decreases decoherence.

    Args:
    ----
    vecset1: A list of vectors
    vecset2: A list of vectors
    backend: Backend on which to run the quantum circuit. Default is
    qiskit_aer.AerSimulator.
    shots: The number times to execute each quantum circuit.
    device_config: An optional config to provide Dask multiprocessing.

    Returns:
    -------
    A matrix of distances from each vector of set1 to that of set1. The row index
    indices the first set index. The column indicates the second set index.

    """
    if not backend:
        backend = AerSimulator()
    distances: np.ndarray = np.ndarray((len(vecset1), len(vecset2)), dtype=np.float64)
    if device_config:
        vecset1, vecset2, norm = build_unit_vectors(vecset1, vecset2)
        vector_size: int = vecset1.shape[1]
        vecset1_partitions: list[np.ndarray] = _create_partitions(
            vecset1,
            device_config.jobs,
        )
        distances = distributed_multi_euclidean(
            vecset1_partitions,
            vecset2,
            vector_size,
            device_config=device_config,
            norm=norm,
            backend=backend,
            shots=shots,
        )
    else:
        for i in range(0, len(vecset1), 2):
            logger.warning(
                "Calculating vectors %s and %s / %s from first vector set",
                str(i),
                str(i + 1),
                str(len(vecset1)),
            )
            try:
                distances[i : i + 2] = multi_euclidean(
                    vecset1[i : i + 2],
                    vecset2,
                    backend=backend,
                    shots=shots,
                )
            except IndexError:
                distances[i] = multi_euclidean(
                    vecset1[i],
                    vecset2,
                    backend=backend,
                    shots=shots,
                )

    return distances


def compare_euclideans(
    vecset1: np.ndarray,
    vecset2: np.ndarray,
    quantum_results: np.ndarray,
) -> float:
    """
    Calculate the Mean Squared Error (MSE) between the quantum and classical results.

    Args:
    ----
    vecset1: A 2D array with vectors as rows.
    vecset2: A 2D array with vectors as rows.
    quantum_results: The distances between vectors, as a square matrix.

    Returns:
    -------
    The MSE for the difference between vectors calculated with different methods.

    """
    error: float = 0
    for i, vec1 in enumerate(vecset1):
        for j, vec2 in enumerate(vecset2):
            error += abs(sqrt(quantum_results[i][j]) - np.linalg.norm(vec1 - vec2))

    return error / (len(vecset1) * len(vecset2))
