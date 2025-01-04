from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from squib import visualization


def test_plot_classical_statevector() -> None:
    results: pd.DataFrame = visualization.load_results(Path("test_files/results.csv"))
    with TemporaryDirectory() as temp_dir:
        visualization.plot_classical_statevector(results, Path(temp_dir))


def test_plot_statevector_aer() -> None:
    results: pd.DataFrame = visualization.load_results(Path("test_files/results.csv"))
    with TemporaryDirectory() as temp_dir:
        visualization.plot_statevector_aer(results)


def test_plot_aer_zardini() -> None:
    results: pd.DataFrame = visualization.load_results(Path("test_files/results.csv"))
    zardini_results: pd.DataFrame = visualization.load_results(
        Path("test_files/zardini.csv"),
    )
    with TemporaryDirectory() as temp_dir:
        visualization.plot_aer_zardini(results, zardini_results)


def test_plot_shots_performance() -> None:
    results: pd.DataFrame = visualization.load_results(Path("test_files/results.csv"))
    with TemporaryDirectory() as temp_dir:
        visualization.plot_shots_performance(results)
