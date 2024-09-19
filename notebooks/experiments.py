"""Script for convenient execution of quantum knn experiments."""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path
from sys import modules
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator, StatevectorSimulator
from ucimlrepo import fetch_ucirepo

from squib.evaluation.knn import cross_validate_knn
from squib.qnn import qnn

if TYPE_CHECKING:
    from argparse import Namespace
    from typing import TextIO

    from qiskit_aer import AerProvider

    from squib.evaluation.metrics import Metrics


logger: logging.Logger = logging.getLogger(__name__)


def parse_script_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(prog="Run KNN and QNN for the UCI Iris dataset")

    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        required=True,
        help="The name of the dataset to run qnn on",
    )

    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_directory",
        type=Path,
        required=False,
        default=Path(),
        help="The directory in which to save the results to",
    )

    parser.add_argument(
        "-l",
        "--log_directory",
        dest="log_directory",
        type=Path,
        required=False,
        default=Path(),
        help="The directory in which to save the output log",
    )

    parser.add_argument(
        "-k",
        dest="k",
        type=int,
        required=False,
        default=3,
        help="The number of nearest neigbhors to check",
    )

    parser.add_argument(
        "-b",
        "--backend",
        dest="backend",
        type=str,
        required=False,
        default="statevectorsimulator",
        help="The quantum or simulated backend on which to run the experiment",
    )

    parser.add_argument(
        "-s",
        "--shots",
        dest="shots",
        type=int,
        required=False,
        default=1024,
        help="The number of times to execute the resultant quantum circuits",
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        required=False,
        default=1,
        help="The seed for the cross-validator",
    )

    return parser.parse_args()


def _save_results(
    file_descriptor: TextIO,
    classical_metrics: list[Metrics],
    quantum_metrics: list[Metrics],
    dataset_name: str,
) -> None:
    """
    Save the metrics to a text file.

    Args:
    ----
    file_descriptor: The open file descriptor to write to
    classical_metrics: The results of a classical run
    quantum_metrics: The results of a quantum run
    dataset_name: The name of the dataset to save results for

    """
    file_descriptor.write(f"{dataset_name.upper()} RESULTS\n")
    file_descriptor.write("CLASSICAL\n")
    for index, metric in enumerate(classical_metrics, start=1):
        file_descriptor.write(f"Fold {index}: {metric}\n")
    file_descriptor.write("\nQUANTUM\n")
    for index, metric in enumerate(quantum_metrics, start=1):
        file_descriptor.write(f"Fold {index}: {metric}\n")
    file_descriptor.write("\n")


def _create_output_filename(
    database_name: str, k: int, backend: AerProvider | StatevectorSimulator, shots: int,
) -> str:
    filename: str = database_name + "_"
    if isinstance(backend, AerSimulator):
        filename += "aer" + str(k) + "_shots" + str(shots)
    elif isinstance(backend, StatevectorSimulator):
        filename += "statevector" + str(k)
    return filename + ".txt"


def iris(
    output_directory: Path = Path(),
    *,
    k: int = 3,
    backend: AerProvider | StatevectorSimulator | None = None,
    shots: int = 1024,
    seed: int = 1,
) -> None:
    """
    Run classical and quantum knn on the UCI ML Iris dataset.

    Args:
    ----
    output_directory: The directory in which to save results to
    k: The k value for the knn algorithm
    backend: The execution medium for the quantum run
    shots: The number of times to execute the quantum circuit
    seed: The seed for the KFold cross validator

    """
    if not backend:
        backend = StatevectorSimulator()
    if not output_directory:
        output_directory = Path()
    transfusion_data: pd.DataFrame = fetch_ucirepo(id=53)
    features: np.ndarray = transfusion_data.data.features.to_numpy(dtype=float)
    targets: np.ndarray = np.squeeze(transfusion_data.data.targets.to_numpy(), axis=1)
    targets[targets == "Iris-setosa"] = 0
    targets[targets == "Iris-versicolor"] = 1
    targets[targets == "Iris-virginica"] = 2
    targets = targets.astype(int)
    indices = np.where(targets <= 1)
    setosa_veriscolor_features: np.ndarray = features[indices[0]]
    setosa_veriscolor_targets: np.ndarray = targets[indices[0]]
    processed_setosa_veriscolor: np.ndarray = qnn.preprocess(setosa_veriscolor_features)
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_setosa_veriscolor,
        setosa_veriscolor_targets,
        k=k,
        seed=seed,
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_setosa_veriscolor,
        setosa_veriscolor_targets,
        k=k,
        backend=backend,
        shots=shots,
        seed=seed,
    )
    filename: str = _create_output_filename("iris", k, backend, shots)
    with (output_directory / filename).open("w") as fd:
        _save_results(fd, classical_metrics, metrics, "SETOSA-VERISCOLOR")

    targets[targets == 1] = 3
    targets[targets == 2] = 1  # noqa: PLR2004
    indices = np.where(targets <= 1)
    setosa_virginica_features: np.ndarray = features[indices[0]]
    setosa_virginica_targets: np.ndarray = targets[indices[0]]
    processed_setosa_virginica: np.ndarray = qnn.preprocess(setosa_virginica_features)
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_setosa_virginica,
        setosa_virginica_targets,
        k=k,
        seed=seed,
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_setosa_virginica,
        setosa_virginica_targets,
        k=k,
        backend=backend,
        shots=shots,
        seed=seed,
    )
    with (output_directory / filename).open("a") as fd:
        _save_results(fd, classical_metrics, metrics, "SETOSA-VIRGINICA")

    targets[targets == 0] = 2
    targets[targets == 3] = 0  # noqa: PLR2004
    indices = np.where(targets <= 1)
    veriscolor_virginica_features: np.ndarray = features[indices[0]]
    veriscolor_virginica_targets: np.ndarray = targets[indices[0]]
    processed_veriscolor_virginica: np.ndarray = qnn.preprocess(
        veriscolor_virginica_features,
    )
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_veriscolor_virginica,
        veriscolor_virginica_targets,
        k=k,
        seed=seed,
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_veriscolor_virginica,
        veriscolor_virginica_targets,
        k=k,
        backend=backend,
        shots=shots,
        seed=seed,
    )
    with (output_directory / filename).open("a") as fd:
        _save_results(fd, classical_metrics, metrics, "VERISCOLOR-VIRGINICA")


def transfusion(
    output_directory: Path = Path(),
    *,
    k: int = 3,
    backend: AerProvider | StatevectorSimulator | None = None,
    shots: int = 1024,
    seed: int = 1,
) -> None:
    """
    Run classical and quantum knn on the UCI ML Transfusion dataset.

    Args:
    ----
    output_directory: The directory in which to save results to
    k: The k value for the knn algorithm
    backend: The execution medium for the quantum run
    shots: The number of times to execute the quantum circuit
    seed: The seed for the KFold cross validator

    """
    if not backend:
        backend = StatevectorSimulator()
    transfusion_data: pd.DataFrame = fetch_ucirepo(id=176)
    features: np.ndarray = transfusion_data.data.features
    targets: np.ndarray = np.squeeze(transfusion_data.data.targets.to_numpy(), axis=1)
    processed_features: np.ndarray = qnn.preprocess(features.to_numpy(float))
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_features,
        targets,
        k=k,
        seed=seed,
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_features,
        targets,
        k=k,
        backend=backend,
        shots=shots,
        seed=seed,
    )

    filename: str = _create_output_filename("transfusion", k, backend, shots)
    with (output_directory / filename).open("w") as fd:
        _save_results(fd, classical_metrics, metrics, "TRANSFUSION")


def vertebral(
    output_directory: Path = Path(),
    *,
    k: int = 3,
    backend: AerProvider | StatevectorSimulator | None = None,
    shots: int = 1024,
    seed: int = 1,
) -> None:
    """
    Run classical and quantum knn on the UCI ML Vertebral Column dataset.

    Args:
    ----
    output_directory: The directory in which to save results to
    k: The k value for the knn algorithm
    backend: The execution medium for the quantum run
    shots: The number of times to execute the quantum circuit
    seed: The seed for the KFold cross validator

    """
    if not backend:
        backend = StatevectorSimulator()
    vertebra_data: pd.DataFrame = fetch_ucirepo(id=212)
    features: np.ndarray = vertebra_data.data.features
    targets: pd.DataFrame | np.ndarray = vertebra_data.data.targets
    targets.loc[targets["class"] == "Normal"] = 0
    targets.loc[targets["class"] != 0] = 1
    targets = np.squeeze(targets.to_numpy(int), axis=1)
    processed_features: np.ndarray = qnn.preprocess(features.to_numpy(float))
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_features,
        targets,
        k=k,
        seed=seed,
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_features,
        targets,
        k=k,
        backend=backend,
        shots=shots,
        seed=seed,
    )

    filename: str = _create_output_filename("vertebral", k, backend, shots)
    with (output_directory / filename).open("w") as fd:
        _save_results(fd, classical_metrics, metrics, "VERTEBRAL_COLUMN")


def ecoli(
    output_directory: Path = Path(),
    *,
    k: int = 3,
    backend: AerProvider | StatevectorSimulator | None = None,
    shots: int = 1024,
    seed: int = 1,
) -> None:
    """
    Run classical and quantum knn on the UCI ML Ecoli dataset.

    Args:
    ----
    output_directory: The directory in which to save results to
    k: The k value for the knn algorithm
    backend: The execution medium for the quantum run
    shots: The number of times to execute the quantum circuit
    seed: The seed for the KFold cross validator

    """
    if not backend:
        backend = StatevectorSimulator()
    ecoli_data: pd.DataFrame = fetch_ucirepo(id=39)
    features: np.ndarray = ecoli_data.data.features.to_numpy(float)
    targets: pd.DataFrame | np.ndarray = ecoli_data.data.targets
    targets.loc[targets["class"] == "cp"] = 0
    targets.loc[targets["class"] == "im"] = 1
    targets.loc[targets["class"] == "imS"] = 2
    targets.loc[targets["class"] == "imL"] = 3
    targets.loc[targets["class"] == "imU"] = 4
    targets.loc[targets["class"] == "om"] = 5
    targets.loc[targets["class"] == "omL"] = 6
    targets.loc[targets["class"] == "pp"] = 7
    targets = np.squeeze(targets.to_numpy(int), axis=1)
    indices = np.where(targets <= 1)
    processed_features: np.ndarray = qnn.preprocess(features[indices[0]])
    targets = targets[indices]
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_features,
        targets,
        k=k,
        seed=seed,
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_features,
        targets,
        k=k,
        backend=backend,
        shots=shots,
        seed=seed,
    )

    filename: str = _create_output_filename("ecoli", k, backend, shots)
    with (output_directory / filename).open("w") as fd:
        _save_results(fd, classical_metrics, metrics, "ECOLI")


def glass(
    output_directory: Path = Path(),
    *,
    k: int = 3,
    backend: AerProvider | StatevectorSimulator | None = None,
    shots: int = 1024,
    seed: int = 1,
) -> None:
    """
    Run classical and quantum knn on the UCI ML Glass dataset.

    Args:
    ----
    output_directory: The directory in which to save results to
    k: The k value for the knn algorithm
    backend: The execution medium for the quantum run
    shots: The number of times to execute the quantum circuit
    seed: The seed for the KFold cross validator

    """
    if not backend:
        backend = StatevectorSimulator()
    vertebra_data: pd.DataFrame = fetch_ucirepo(id=42)
    features: np.ndarray = vertebra_data.data.features.to_numpy(float)
    targets: np.ndarray = np.squeeze(
        vertebra_data.data.targets.to_numpy(),
        axis=1,
    ).astype(int)
    targets[targets == 0] = 8
    targets[targets == 2] = 0  # noqa: PLR2004
    indices = np.where(targets <= 1)
    processed_features: np.ndarray = qnn.preprocess(features[indices])
    print(targets[indices])
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_features,
        targets[indices],
        k=k,
        seed=seed,
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_features,
        targets[indices],
        k=k,
        backend=backend,
        shots=shots,
        seed=seed,
    )

    filename: str = _create_output_filename("glass", k, backend, shots)
    with (output_directory / filename).open("w") as fd:
        _save_results(fd, classical_metrics, metrics, "GLASS")


def breast_cancer(
    output_directory: Path = Path(),
    *,
    k: int = 3,
    backend: AerProvider | StatevectorSimulator | None = None,
    shots: int = 1024,
    seed: int = 1,
) -> None:
    """
    Run classical and quantum knn on the UCI ML Breast Cancer dataset.

    Args:
    ----
    output_directory: The directory in which to save results to
    k: The k value for the knn algorithm
    backend: The execution medium for the quantum run
    shots: The number of times to execute the quantum circuit
    seed: The seed for the KFold cross validator

    """
    if not backend:
        backend = StatevectorSimulator()
    vertebra_data: pd.DataFrame = fetch_ucirepo(id=451)
    features: np.ndarray = vertebra_data.data.features.to_numpy(float)
    targets: np.ndarray = np.squeeze(
        vertebra_data.data.targets.to_numpy(),
        axis=1,
    ).astype(int)
    targets[targets == 1] = 0
    targets[targets == 2] = 1  # noqa: PLR2004
    processed_features: np.ndarray = qnn.preprocess(features)
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_features,
        targets,
        k=k,
        seed=seed,
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_features,
        targets,
        k=k,
        backend=backend,
        shots=shots,
        seed=seed,
    )

    filename: str = _create_output_filename("breast_cancer", k, backend, shots)
    with (output_directory / filename).open("w") as fd:
        _save_results(fd, classical_metrics, metrics, "BREAST_CANCER")


def parse_backend(backend_name: str) -> AerProvider | StatevectorSimulator:
    """
    Parse a backend string to return that backend.

    Args:
    ----
    backend_name: The name of the backend to retrieve

    """
    if backend_name.lower() == "aersimulator":
        return AerSimulator()

    if backend_name.lower() == "statevectorsimulator":
        return StatevectorSimulator()

    raise ValueError("Backend %s not supported", backend_name)  # noqa: TRY003


if __name__ == "__main__":
    args: Namespace = parse_script_args()
    logging.basicConfig(
        filename=args.log_directory / f"{args.dataset}_experiment_results{args.k}.log",
        level=logging.WARNING,
    )
    function = getattr(modules[__name__], args.dataset)
    if function is None:
        raise ValueError("Dataset %s is not supported", args.dataset)  # noqa: TRY003

    backend = parse_backend(args.backend)
    function(
        args.output_directory,
        k=args.k,
        backend=backend,
        shots=args.shots,
        seed=args.seed,
    )
