import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from squib.evaluation import parse_zardini


def setup_file_structure(top_directory: Path) -> None:
    classical_directory: Path = top_directory / "classical"
    classical_directory.mkdir()
    (top_directory / "statevector").mkdir()
    (top_directory / "simulation").mkdir()
    (top_directory / "extra_file.json").touch()
    subtype_directory: Path = classical_directory / "classical"
    subtype_directory.mkdir()
    dataset_directory: Path = subtype_directory / "01_iris_setosa_versicolor"
    dataset_directory.mkdir()
    (subtype_directory / "06_transfusion").mkdir()
    k_directory: Path = dataset_directory / "k_3"
    k_directory.mkdir()
    (dataset_directory / "k_5").mkdir()
    run_directory: Path = k_directory / "run_0"
    run_directory.mkdir()
    (dataset_directory / "k_5/run_0").mkdir()
    shutil.copy("test_files/results_processed.json", run_directory)
    (k_directory / "extra_file.json").touch()
    shutil.copy("test_files/results_processed.json", dataset_directory / "k_5/run_0")


def test_process_results_file() -> None:
    results_details: dict[str, float] = {}
    parse_zardini.process_results_file(
        Path("test_files/results_processed.json"),
        results_details,
    )
    assert results_details["accuracy"] == pytest.approx(0.7949425287356322)
    assert results_details["jaccard"] == 1.0
    assert results_details["average_jaccard"] == 1.0


def test_process_k_directory() -> None:
    results_details: dict[str, str | int] = {}
    with TemporaryDirectory() as temp_dir:
        setup_file_structure(Path(temp_dir))
        parse_zardini.process_k_directory(
            Path(temp_dir) / "classical/classical/01_iris_setosa_versicolor/k_3",
            results_details,
        )
        assert results_details["accuracy"] == pytest.approx(0.7949425287356322)
        assert results_details["jaccard"] == 1.0
        assert results_details["average_jaccard"] == 1.0


def test_process_dataset_directory() -> None:
    results_details: dict[str, str | int] = {}
    with TemporaryDirectory() as temp_dir:
        setup_file_structure(Path(temp_dir))
        results: list[dict[str, str | int | float]] = (
            parse_zardini.process_dataset_directory(
                Path(temp_dir) / "classical/classical/01_iris_setosa_versicolor",
                results_details,
            )
        )
        assert results[0]["accuracy"] == pytest.approx(0.7949425287356322)
        assert results[0]["jaccard"] == 1.0
        assert results[0]["average_jaccard"] == 1.0


def test_process_submethod_directory() -> None:
    results_details: dict[str, str | int] = {}
    with TemporaryDirectory() as temp_dir:
        setup_file_structure(Path(temp_dir))
        results: list[dict[str, str | int | float]] = (
            parse_zardini.process_submethod_directory(
                Path(temp_dir) / "classical/classical",
                results_details,
            )
        )
        assert results[0]["accuracy"] == pytest.approx(0.7949425287356322)
        assert results[0]["jaccard"] == 1.0
        assert results[0]["average_jaccard"] == 1.0


def test_process_execution_method() -> None:
    with TemporaryDirectory() as temp_dir:
        setup_file_structure(Path(temp_dir))
        results: list[dict[str, str | int | float]] = (
            parse_zardini.process_execution_method(Path(temp_dir) / "classical")
        )
        assert results[0]["accuracy"] == pytest.approx(0.7949425287356322)
        assert results[0]["jaccard"] == 1.0
        assert results[0]["average_jaccard"] == 1.0


def test_process_results() -> None:
    with TemporaryDirectory() as temp_dir:
        setup_file_structure(Path(temp_dir))
        results: pd.DataFrame = parse_zardini.process_results(Path(temp_dir))
        assert results.iloc[0]["accuracy"] == pytest.approx(0.7949425287356322)
        assert results.iloc[0]["jaccard"] == 1.0
        assert results.iloc[0]["average_jaccard"] == 1.0
