from __future__ import annotations

from argparse import ArgumentParser
import logging
from pathlib import Path
from sys import modules
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from qiskit_aer import StatevectorSimulator

from squib.qnn import qnn
from squib.evaluation.knn import cross_validate_knn

if TYPE_CHECKING:
    from argparse import Namespace

    from squib.evaluation.metrics import Metrics


logger: logging.Logger = logging.getLogger(__name__)


def parse_script_args() -> Namespace:
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

    return parser.parse_args()


def save_results(
    file_descriptor,
    classical_metrics: list[Metrics],
    quantum_metrics: list[Metrics],
    dataset_name: str,
) -> None:
    file_descriptor.write(f"{dataset_name.upper()} RESULTS\n")
    file_descriptor.write("CLASSICAL\n")
    for index, metric in enumerate(classical_metrics, start=1):
        file_descriptor.write(f"Fold {index}: {metric}\n")
    file_descriptor.write("\nQUANTUM\n")
    for index, metric in enumerate(quantum_metrics, start=1):
        file_descriptor.write(f"Fold {index}: {metric}\n")
    file_descriptor.write("\n")


def iris(output_directory: Path = None) -> None:
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
        processed_setosa_veriscolor, targets
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_setosa_veriscolor,
        setosa_veriscolor_targets,
        backend=backend,
    )
    with (output_directory / "iris.txt").open("w") as fd:
        save_results(fd, classical_metrics, metrics, "SETOSA-VERISCOLOR")

    targets[targets == 1] = 3
    targets[targets == 2] = 1
    indices = np.where(targets <= 1)
    setosa_virginica_features: np.ndarray = features[indices[0]]
    setosa_virginica_targets: np.ndarray = targets[indices[0]]
    processed_setosa_virginica: np.ndarray = qnn.preprocess(setosa_virginica_features)
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_setosa_virginica, targets
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_setosa_virginica,
        setosa_virginica_targets,
        backend=backend,
    )
    with (output_directory / "iris.txt").open("a") as fd:
        save_results(fd, classical_metrics, metrics, "SETOSA-VIRGINICA")

    targets[targets == 0] = 2
    targets[targets == 3] = 0
    indices = np.where(targets <= 1)
    veriscolor_virginica_features: np.ndarray = features[indices[0]]
    veriscolor_virginica_targets: np.ndarray = targets[indices[0]]
    processed_veriscolor_virginica: np.ndarray = qnn.preprocess(
        veriscolor_virginica_features
    )
    classical_metrics: list[Metrics] = cross_validate_knn(
        processed_veriscolor_virginica, targets
    )
    metrics: list[Metrics] = qnn.cross_validate(
        processed_veriscolor_virginica,
        veriscolor_virginica_targets,
        backend=backend,
    )
    with (output_directory / "iris.txt").open("a") as fd:
        save_results(fd, classical_metrics, metrics, "VERISCOLOR-VIRGINICA")


def transfusion(output_directory: Path = None) -> None:
    if not output_directory:
        output_directory = Path()
    transfusion_data: pd.DataFrame = fetch_ucirepo(id=176)
    features: np.ndarray = transfusion_data.data.features
    targets: np.ndarray = np.squeeze(transfusion_data.data.targets.to_numpy(), axis=1)
    processed_features: np.ndarray = qnn.preprocess(features.to_numpy(float))
    classical_metrics: list[Metrics] = cross_validate_knn(processed_features, targets)
    metrics: list[Metrics] = qnn.cross_validate(processed_features, targets)

    with (output_directory / "transfusion.txt").open("w") as fd:
        save_results(fd, classical_metrics, metrics, "TRANSFUSION")


def vertebral(output_directory: Path = None) -> None:
    if not output_directory:
        output_directory = Path()
    vertebra_data: pd.DataFrame = fetch_ucirepo(id=212)
    features: np.ndarray = vertebra_data.data.features
    targets: pd.DataFrame | np.ndarray = vertebra_data.data.targets
    targets.loc[targets["class"] == "Normal"] = 0
    targets.loc[targets["class"] != 0] = 1
    targets = np.squeeze(targets.to_numpy(int), axis=1)
    processed_features: np.ndarray = qnn.preprocess(features.to_numpy(float))
    classical_metrics: list[Metrics] = cross_validate_knn(processed_features, targets)
    metrics: list[Metrics] = qnn.cross_validate(processed_features, targets)

    with (output_directory / "vertebral_column.txt").open("w") as fd:
        save_results(fd, classical_metrics, metrics, "VERTEBRAL_COLUMN")


def ecoli(output_directory: Path = None) -> None:
    if not output_directory:
        output_directory = Path()
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
    processed_features: np.ndarray = qnn.preprocess(
        features[indices[0]]
    )
    targets = targets[indices]
    classical_metrics: list[Metrics] = cross_validate_knn(processed_features, targets)
    metrics: list[Metrics] = qnn.cross_validate(processed_features, targets)

    with (output_directory / "ecoli.txt").open("w") as fd:
        save_results(fd, classical_metrics, metrics, "ECOLI")


def glass(output_directory: Path = None) -> None:
    if not output_directory:
        output_directory = Path()
    vertebra_data: pd.DataFrame = fetch_ucirepo(id=42)
    features: np.ndarray = vertebra_data.data.features.to_numpy(float)
    targets: np.ndarray = np.squeeze(
        vertebra_data.data.targets.to_numpy(), axis=1
    ).astype(int)
    targets[targets == 0] = 8
    targets[targets == 2] = 1
    indices = np.where(targets <= 1)
    processed_features: np.ndarray = qnn.preprocess(
        features[indices]
    )
    classical_metrics: list[Metrics] = cross_validate_knn(processed_features, targets)
    metrics: list[Metrics] = qnn.cross_validate(processed_features, targets)

    with (output_directory / "glass.txt").open("w") as fd:
        save_results(fd, classical_metrics, metrics, "GLASS")


def breast_cancer(output_directory: Path = None) -> None:
    if not output_directory:
        output_directory = Path()
    vertebra_data: pd.DataFrame = fetch_ucirepo(id=451)
    features: np.ndarray = vertebra_data.data.features.to_numpy(float)
    targets: np.ndarray = np.squeeze(
        vertebra_data.data.targets.to_numpy(), axis=1
    ).astype(int)
    targets[targets == 1] = 0
    targets[targets == 2] = 1
    processed_features: np.ndarray = qnn.preprocess(features)
    classical_metrics: list[Metrics] = cross_validate_knn(processed_features, targets)
    metrics: list[Metrics] = qnn.cross_validate(processed_features, targets)

    with (output_directory / "breast_cancer.txt").open("w") as fd:
        save_results(fd, classical_metrics, metrics, "BREAST_CANCER")


if __name__ == "__main__":
    args: Namespace = parse_script_args()
    logging.basicConfig(
        filename=args.log_directory / f"{args.dataset}_experiment_results.log",
        level=logging.WARNING,
    )
    function = getattr(modules[__name__], args.dataset)
    if function == None:
        raise ValueError("Dataset %s is not supported", args.dataset)
    function(args.output_directory)
