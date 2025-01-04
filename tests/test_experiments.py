from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from notebooks import experiments


@pytest.mark.parametrize(
    ("backend_name"),
    ["aersimulator", "fake", "statevectorsimulator", "ibm_quantum"],
)
@mock.patch("notebooks.experiments._setup_ibm_runtime")
def test_parse_backend(mock_setup: mock.Mock, backend_name: str) -> None:
    mock_setup.return_value = "ibm_backend"
    if backend_name != "ibm_quantum":
        backend = experiments._parse_backend(backend_name)
    else:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            token_filepath: Path = temp_path / "token"
            token_filepath.touch()
            backend = experiments._parse_backend(backend_name, token_filepath)

    if backend_name == "aersimulator":
        assert isinstance(backend, AerSimulator)
    if backend_name == "fake":
        assert isinstance(backend, FakeBrisbane)
    elif backend_name == "statevectorsimulator":
        assert isinstance(backend, StatevectorSimulator)
    elif backend_name == "ibm_quantum":
        assert backend == "ibm_backend"
