"""Visualizations comparing results between algorithms and execution methods."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_results(results_filepath: Path) -> pd.DataFrame:
    """
    Load results into a DataFrame.

    Args:
    ----
    results_filepath: The filepath to the results file.

    Returns:
    -------
    DataFrame: The results data.

    """
    return pd.read_csv(
        results_filepath,
        dtype={"accuracy": float, "jaccard": float, "average_jaccard": float},
    )


def plot_classical_statevector(results: pd.DataFrame, output_directory: Path) -> None:
    """
    Plot the classical vs statevector execution results, saving them as PNG files.

    Args:
    ----
    results: The results data.
    output_directory: The directory in which to save the figures.

    """
    border_limit: float = 1.05
    plt.clf()
    plt.xlim([0.0, border_limit])
    plt.ylim([0.0, border_limit])
    for k, color, marker in zip(
        [3, 5, 7, 9],
        ["b", "g", "r", "k"],
        ["x", "o", "*", "s"],
    ):
        k_subset: pd.DataFrame = results[results["k"] == k]
        plt.scatter(
            k_subset[k_subset["execution"] == "classical"]["accuracy"],
            k_subset[k_subset["execution"] == "statevector"]["accuracy"],
            c=color,
            s=50,
            marker=marker,
            label=f"k = {k}",
        )
    plt.plot([0.0, border_limit], [0.0, border_limit], "b--", alpha=0.25)
    plt.legend()
    plt.xlabel("Average Classical Accuracy")
    plt.ylabel("Average Statevector Accuracy")
    plt.savefig(output_directory / "classical_statevector_accuracy.png")

    plt.clf()
    plt.xlim([0.0, border_limit])
    plt.ylim([0.0, border_limit])
    for k, color, marker in zip(
        [3, 5, 7, 9],
        ["b", "g", "r", "k"],
        ["x", "o", "*", "s"],
    ):
        k_subset: pd.DataFrame = results[results["k"] == k]
        plt.scatter(
            k_subset[k_subset["execution"] == "classical"]["jaccard"],
            k_subset[k_subset["execution"] == "statevector"]["jaccard"],
            c=color,
            marker=marker,
            label=f"k = {k}",
        )
    plt.plot([0.0, border_limit], [0.0, border_limit], "b--", alpha=0.25)
    plt.legend()
    plt.xlabel("Average Classical Jaccard Index")
    plt.ylabel("Average Statevector Jaccard Index")
    plt.savefig(output_directory / "classical_statevector_ji.png")

    plt.clf()
    plt.xlim([0.0, border_limit])
    plt.ylim([0.0, border_limit])
    for k, color, marker in zip(
        [3, 5, 7, 9],
        ["b", "g", "r", "k"],
        ["x", "o", "*", "s"],
    ):
        k_subset: pd.DataFrame = results[results["k"] == k]
        plt.scatter(
            k_subset[k_subset["execution"] == "classical"]["average_jaccard"],
            k_subset[k_subset["execution"] == "statevector"]["average_jaccard"],
            c=color,
            marker=marker,
            label=f"k = {k}",
        )
    plt.plot([0.0, border_limit], [0.0, border_limit], "b--", alpha=0.25)
    plt.legend()
    plt.xlabel("Average Classical Average Jaccard Index")
    plt.ylabel("Average Statevector Average Jaccard Index")
    plt.savefig(output_directory / "classical_statevector_aji.png")


def plot_statevector_aer(
    results: pd.DataFrame,
    shots: int = 1024,
    output_directory: Path = Path(),
) -> None:
    """
    Plot the statevector vs AerSimulator execution results, saving them as PNG files.

    Args:
    ----
    results: The results data.
    shots: The shots to plot.
    output_directory: The directory in which to save the figures.

    """
    border_limit: float = 1.05
    plt.clf()
    plt.xlim([0.0, border_limit])
    plt.ylim([0.0, border_limit])
    for k, color, marker in zip(
        [3, 5, 7, 9],
        ["b", "g", "r", "k"],
        ["x", "o", "*", "s"],
    ):
        k_subset: pd.DataFrame = results[results["k"] == k]
        aer_subset: pd.DataFrame = k_subset[k_subset["shots"] == shots]
        plt.scatter(
            k_subset[k_subset["execution"] == "statevector"]["accuracy"],
            aer_subset[aer_subset["execution"] == "aer"]["accuracy"],
            c=color,
            marker=marker,
            label=f"k = {k}",
        )
    plt.plot([0.0, border_limit], [0.0, border_limit], "b--", alpha=0.25)
    plt.legend()
    plt.xlabel("Average Statevector Accuracy")
    plt.ylabel("Average Aer Simulator Accuracy")
    plt.savefig(output_directory / "statevector_aer_accuracy.png")

    plt.clf()
    plt.xlim([0.0, border_limit])
    plt.ylim([0.0, border_limit])
    for k, color, marker in zip(
        [3, 5, 7, 9],
        ["b", "g", "r", "k"],
        ["x", "o", "*", "s"],
    ):
        k_subset: pd.DataFrame = results[results["k"] == k]
        aer_subset: pd.DataFrame = k_subset[k_subset["shots"] == shots]
        plt.scatter(
            k_subset[k_subset["execution"] == "statevector"]["jaccard"],
            aer_subset[aer_subset["execution"] == "aer"]["jaccard"],
            c=color,
            marker=marker,
            label=f"k = {k}",
        )
    plt.plot([0.0, border_limit], [0.0, border_limit], "b--", alpha=0.25)
    plt.legend()
    plt.xlabel("Average Statevector Jaccard Index")
    plt.ylabel("Average Aer Simulator Jaccard Index")
    plt.savefig(output_directory / "statevector_aer_ji.png")

    plt.clf()
    plt.xlim([0.0, border_limit])
    plt.ylim([0.0, border_limit])
    for k, color, marker in zip(
        [3, 5, 7, 9],
        ["b", "g", "r", "k"],
        ["x", "o", "*", "s"],
    ):
        k_subset: pd.DataFrame = results[results["k"] == k]
        aer_subset: pd.DataFrame = k_subset[k_subset["shots"] == shots]
        plt.scatter(
            k_subset[k_subset["execution"] == "statevector"]["average_jaccard"],
            aer_subset[aer_subset["execution"] == "aer"]["average_jaccard"],
            c=color,
            marker=marker,
            label=f"k = {k}",
        )
    plt.plot([0.0, border_limit], [0.0, border_limit], "b--", alpha=0.25)
    plt.legend()
    plt.xlabel("Average Statevector Average Jaccard Index")
    plt.ylabel("Average Aer Simulator Average Jaccard Index")
    plt.savefig(output_directory / "statevector_aer_aji.png")


def plot_aer_zardini(  # noqa: PLR0915
    results: pd.DataFrame,
    zardini_results: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Plot the statevector vs AerSimulator execution results, saving them as PNG files.

    Args:
    ----
    results: The results data.
    zardini_results: The results from the Zardini et. al. algorithm.
    output_directory: The directory in which to save the figures.

    """
    border_limit: float = 1.05
    plt.clf()
    plt.xlim([0.0, border_limit])
    plt.ylim([0.0, border_limit])
    dataset_names: list[str] = results["dataset"].unique()
    zardini_results = zardini_results[zardini_results["dataset"].isin(dataset_names)]
    for k, color, marker in zip(
        [3, 5, 7, 9],
        ["b", "g", "r", "k"],
        ["x", "o", "*", "s"],
    ):
        k_subset: pd.DataFrame = results[results["k"] == k]
        aer_subset: pd.DataFrame = k_subset[k_subset["shots"] == 8192]  # noqa: PLR2004
        zardini_k_subset: pd.DataFrame = zardini_results[zardini_results["k"] == k]
        aer_plot_data = aer_subset[aer_subset["execution"] == "aer"]
        zardini_plot_data = zardini_k_subset[
            zardini_k_subset["execution"] == "local_simulation"
        ]
        dataset_names: np.ndarray = zardini_plot_data["dataset"].unique()
        zardini_plot_data = zardini_plot_data.set_index("dataset").sort_index()
        aer_plot_data = (
            aer_plot_data[aer_plot_data["dataset"].isin(dataset_names)]
            .set_index("dataset")
            .sort_index()
        )
        plt.scatter(
            x=zardini_plot_data["accuracy"],
            y=aer_plot_data["accuracy"],
            c=color,
            marker=marker,
            label=f"k = {k}",
        )
    plt.plot([0.0, border_limit], [0.0, border_limit], "b--", alpha=0.25)
    plt.legend()
    plt.xlabel("Average Extension Simulation Accuracy")
    plt.ylabel("Average Aer Simulator Accuracy")
    plt.savefig(output_directory / "zardini_aer_accuracy.png")

    plt.clf()
    plt.xlim([0.0, border_limit])
    plt.ylim([0.0, border_limit])
    for k, color, marker in zip(
        [3, 5, 7, 9],
        ["b", "g", "r", "k"],
        ["x", "o", "*", "s"],
    ):
        k_subset: pd.DataFrame = results[results["k"] == k]
        aer_subset: pd.DataFrame = k_subset[k_subset["shots"] == 1024]  # noqa: PLR2004
        zardini_k_subset: pd.DataFrame = zardini_results[zardini_results["k"] == k]
        aer_plot_data = aer_subset[aer_subset["execution"] == "aer"]
        zardini_plot_data = zardini_k_subset[
            zardini_k_subset["execution"] == "local_simulation"
        ]
        dataset_names: np.ndarray = zardini_plot_data["dataset"].unique()
        zardini_plot_data = zardini_plot_data.set_index("dataset").sort_index()
        aer_plot_data = (
            aer_plot_data[aer_plot_data["dataset"].isin(dataset_names)]
            .set_index("dataset")
            .sort_index()
        )
        plt.scatter(
            x=zardini_plot_data["jaccard"],
            y=aer_plot_data["jaccard"],
            c=color,
            marker=marker,
            label=f"k = {k}",
        )
    plt.plot([0.0, border_limit], [0.0, border_limit], "b--", alpha=0.25)
    plt.legend()
    plt.xlabel("Average Extension Simulation Jaccard Index")
    plt.ylabel("Average Aer Simulator Jaccard Index")
    plt.savefig(output_directory / "zardini_aer_ji.png")

    plt.clf()
    plt.xlim([0.0, border_limit])
    plt.ylim([0.0, border_limit])
    for k, color, marker in zip(
        [3, 5, 7, 9],
        ["b", "g", "r", "k"],
        ["x", "o", "*", "s"],
    ):
        k_subset: pd.DataFrame = results[results["k"] == k]
        aer_subset: pd.DataFrame = k_subset[k_subset["shots"] == 1024]  # noqa: PLR2004
        zardini_k_subset: pd.DataFrame = zardini_results[zardini_results["k"] == k]
        aer_plot_data = aer_subset[aer_subset["execution"] == "aer"]
        zardini_plot_data = zardini_k_subset[
            zardini_k_subset["execution"] == "local_simulation"
        ]
        dataset_names: np.ndarray = zardini_plot_data["dataset"].unique()
        zardini_plot_data = zardini_plot_data.set_index("dataset").sort_index()
        aer_plot_data = (
            aer_plot_data[aer_plot_data["dataset"].isin(dataset_names)]
            .set_index("dataset")
            .sort_index()
        )
        plt.scatter(
            x=zardini_plot_data["average_jaccard"],
            y=aer_plot_data["average_jaccard"],
            c=color,
            marker=marker,
            label=f"k = {k}",
        )
    plt.plot([0.0, border_limit], [0.0, border_limit], "b--", alpha=0.25)
    plt.legend()
    plt.xlabel("Average Extension Simulation Average Jaccard Index")
    plt.ylabel("Average Aer Simulator Average Jaccard Index")
    plt.savefig(output_directory / "zardini_aer_aji.png")


def plot_shots_performance(results: pd.DataFrame, output_directory: Path) -> None:
    """
    Save a box-and-whisker plot of performance per shot count.

    Args:
    ----
    results: The results data.
    output_directory: The directory in which to save the figure.

    """
    shots: list[int] = [1024, 2048, 4096, 8192]
    plt.clf()
    aer_subset: pd.DataFrame = results[results["execution"] == "aer"]
    plt.boxplot(
        [
            aer_subset[aer_subset["shots"] == shot_quantity]["accuracy"]
            for shot_quantity in shots
        ],
        vert=True,
        labels=[str(shot_quantity) for shot_quantity in shots],
    )
    plt.xlabel("Shots")
    plt.ylabel("Aer Simulation Accuracy")
    plt.savefig(output_directory / "shots_accuracy.png")

    plt.clf()
    aer_subset: pd.DataFrame = results[results["execution"] == "aer"]
    plt.boxplot(
        [
            aer_subset[aer_subset["shots"] == shot_quantity]["average_jaccard"]
            for shot_quantity in shots
        ],
        vert=True,
        labels=[str(shot_quantity) for shot_quantity in shots],
    )
    plt.xlabel("Shots")
    plt.ylabel("Aer Simulation Average Jaccard Index")
    plt.savefig(output_directory / "shots_aji.png")
