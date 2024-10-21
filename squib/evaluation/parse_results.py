"""Read text results from a quantum knn run."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TextIO


def parse_module_args() -> Namespace:
    """
    Parse the command line arguments.

    Returns
    -------
    The parsed command line arguments

    """
    parser = ArgumentParser("results_parser")
    parser.add_argument(
        "-r",
        "--results",
        dest="results_filepath",
        required=False,
        type=Path,
    )
    parser.add_argument(
        "-f",
        "--folds",
        dest="fold_quantity",
        required=False,
        type=int,
        default=5,
    )
    return parser.parse_args()


def read_one_classical_result(fd: TextIO) -> float:
    """
    Read the metrics from one result.

    Args:
    ----
    fd: The file descriptor for the results file

    Returns:
    -------
    The accuracy, jaccard, average_jaccard

    """
    current_line: list[str] = fd.readline().split(":")
    accuracy = float(current_line[-1])
    for _ in range(4):
        fd.readline()
    return accuracy


def read_one_quantum_result(fd: TextIO) -> tuple[float, float, float]:
    """
    Read the metrics from one result.

    Args:
    ----
    fd: The file descriptor for the results file

    Returns:
    -------
    The accuracy, jaccard, average_jaccard

    """
    current_line: list[str] = fd.readline().split(":")
    accuracy = float(current_line[-1])
    current_line = fd.readline().split(":")
    jaccard: float = float(current_line[-1])
    current_line = fd.readline().split(":")
    average_jaccard: float = float(current_line[-1])
    for _ in range(4):
        fd.readline()
    return accuracy, jaccard, average_jaccard


def parse_results(results_filepath: Path, fold_quantity: int = 5) -> None:
    """
    Parse the results of a quantum knn run text file.

    Args:
    ----
    results_filepath: The filepath for the results text file.
    fold_quantity: The number of folds used in the cross validation.

    """
    mean_accuracy: float = 0
    with results_filepath.open() as fd:
        # Read first two classical headers
        fd.readline()
        fd.readline()
        for _ in range(fold_quantity):
            accuracy = read_one_classical_result(fd)
            mean_accuracy += accuracy
        print(f"Mean classical accuracy: {mean_accuracy / fold_quantity}")

        mean_accuracy = 0
        mean_jaccard: float = 0
        mean_average_jaccard: float = 0
        # Skip blank line
        fd.readline()
        # Read quantum header
        fd.readline()
        for _ in range(fold_quantity):
            accuracy, jaccard, average_jaccard = read_one_quantum_result(fd)
            mean_accuracy += accuracy
            mean_jaccard += jaccard
            mean_average_jaccard += average_jaccard
        print(f"Mean quantum accuracy: {mean_accuracy / fold_quantity}")
        print(f"Mean quantum jaccard index: {mean_jaccard / fold_quantity}")
        print(
            f"Mean quantum average jaccard index: "
            f"{mean_average_jaccard / fold_quantity}",
        )


if __name__ == "__main__":
    args: Namespace = parse_module_args()
    parse_results(args.results_filepath, args.fold_quantity)
