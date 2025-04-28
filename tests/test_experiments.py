from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
from qiskit_aer import AerSimulator, StatevectorSimulator

from notebooks import experiments


@pytest.mark.parametrize(
    "backend_name",
    ["aersimulator", "statevectorsimulator"]
)
def test_parse_backend(backend_name: str) -> None:
    backend = experiments._parse_backend(backend_name)

    if backend_name == "aersimulator":
        assert isinstance(backend, AerSimulator)
    elif backend_name == "statevectorsimulator":
        assert isinstance(backend, StatevectorSimulator)
