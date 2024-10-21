"""Parse a directory of results from running the zardini et. al. algorithm."""

import json
import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


def process_results_file(
    filepath: Path,
    result_details: dict[str, str | int | float],
) -> bool:
    """
    Parse results from one file.

    Args:
    ----
    filepath: The filepath to the results file.
    result_details: A pre-filled dictionary containing the run specifications.

    Returns:
    -------
    bool: True if successful, false if not.

    """
    try:
        with filepath.open() as fd:
            data: dict = json.load(fd)
            try:
                result_details["accuracy"] = float(data["exact"]["accuracy"]["mean"])
                result_details["jaccard"] = float(
                    data["exact"]["jaccard_index"]["mean"],
                )
                result_details["average_jaccard"] = float(
                    data["exact"]["average_jaccard_index"]["mean"],
                )
            except KeyError:
                result_details["accuracy"] = float(data["avg"]["accuracy"]["mean"])
                result_details["jaccard"] = float(data["avg"]["jaccard_index"]["mean"])
                result_details["average_jaccard"] = float(
                    data["avg"]["average_jaccard_index"]["mean"],
                )
    except FileNotFoundError:
        logger.warning("Not found: %s", str(filepath))
        return False

    return True


def process_k_directory(
    directory_path: Path,
    result_details: dict[str, str | int],
) -> bool:
    """
    Process all results in the k-value subdirectory.

    Args:
    ----
    directory_path: The filepath to the k-value subdirectory.
    result_details: A pre-filled dictionary containing the run specifications.

    Returns:
    -------
    bool: True if successful, false if not.

    """
    run_directory: Path = directory_path / "run_0"
    k = int(directory_path.name.split("_")[1])
    result_details["k"] = k
    return process_results_file(
        run_directory / "results_processed.json",
        result_details,
    )


def process_dataset_directory(
    directory_path: Path,
    result_details: dict[str, str | int],
) -> list[dict[str, str | int | float]]:
    """
    Process all results in the dataset subdirectory.

    Args:
    ----
    directory_path: The filepath to the dataset subdirectory.
    result_details: A pre-filled dictionary containing the run specifications.

    Returns:
    -------
    A list of results_details dictionaries.

    """
    dataset = "_".join(directory_path.name.split("_")[1:])
    result_details["dataset"] = dataset
    results: list[dict[str, str | int | float]] = []
    for k_directory in directory_path.iterdir():
        current_details: dict[str, str | int | float] = deepcopy(result_details)
        if process_k_directory(k_directory, current_details):
            results.append(current_details)

    return results


def process_submethod_directory(
    directory_path: Path,
    result_details: dict[str, str | int | float],
) -> list[dict[str, str | int | float]]:
    """
    Process all results in the submethod subdirectory.

    Args:
    ----
    directory_path: The filepath to the submethod subdirectory.
    result_details: A pre-filled dictionary containing the run specifications.

    Returns:
    -------
    A list of results_details dictionaries.

    """
    submethod: str = directory_path.name
    result_details["submethod"] = submethod
    results: list[dict[str, str | int | float]] = []
    for dataset_directory in directory_path.iterdir():
        results += process_dataset_directory(dataset_directory, result_details)

    return results


def process_execution_method(
    directory_path: Path,
) -> list[dict[str, str | int | float]]:
    """
    Process all results in the execution method subdirectory.

    Args:
    ----
    directory_path: The filepath to the dataset subdirectory.

    Returns:
    -------
    A list of results_details dictionaries.

    """
    execution_method: str = directory_path.name
    results_details: dict[str, str | int | float] = {"execution": execution_method}
    results: list[dict[str, str | int | float]] = []
    for submethod_directory in directory_path.iterdir():
        results += process_submethod_directory(submethod_directory, results_details)

    return results


def convert_headers(results: pd.DataFrame) -> None:
    """
    Convert the headers to names compatible with qnn results.

    Args:
    ----
    results: The parsed results data

    """
    results.loc[results["dataset"] == "iris_setosa_virginica", "dataset"] = (
        "setosa_virginica"
    )
    results.loc[results["dataset"] == "iris_setosa_versicolor", "dataset"] = (
        "setosa_versicolor"
    )
    results.loc[results["dataset"] == "iris_versicolor_virginica", "dataset"] = (
        "versicolor_virginica"
    )
    results.loc[results["dataset"] == "ecoli_cp_im", "dataset"] = "ecoli"
    results.loc[results["dataset"] == "glasses_1_2", "dataset"] = "glass"
    results.loc[results["dataset"] == "vertebral_column_2C", "dataset"] = (
        "vertebral_column"
    )


def process_results(directory_path: Path) -> pd.DataFrame:
    """
    Process all Zardini et. al. algorithm results in a directory.

    Args:
    ----
    directory_path: The filepath to the top-level results directory.

    Returns:
    -------
    pd.DataFrame: The parsed results data

    """
    results: list[dict[str, str | int | float]] = []
    for execution_directory in directory_path.iterdir():
        if execution_directory.is_dir():
            results += process_execution_method(execution_directory)

    data = pd.DataFrame(results)
    convert_headers(data)

    return data
