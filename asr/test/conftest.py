import pytest
import os
import numpy as np

from ase import Atoms
from ase.build import bulk


@pytest.fixture
def usemocks(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(Path(__file__).parent.resolve() / "mocks")


# Make some 1D, 2D and 3D test materials
C = bulk("Si")
abn = 2.51
BN = Atoms(
    "BN",
    scaled_positions=[[0, 0, 0], [1 / 3, 2 / 3, 0]],
    cell=[
        [abn, 0.0, 0.0],
        [-0.5 * abn, np.sqrt(3) / 2 * abn, 0],
        [0.0, 0.0, 15.0],
    ],
    pbc=[True, True, False],
)
Agchain = Atoms(
    "Ag",
    scaled_positions=[[0.5, 0.5, 0]],
    cell=[
        [15.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 2],
    ],
    pbc=[False, False, True],
)
test_materials = [C, BN, Agchain]


@pytest.fixture()
def isolated_filesystem(tmpdir):
    """A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    cwd = os.getcwd()
    os.chdir(str(tmpdir))
    try:
        yield
    finally:
        os.chdir(cwd)


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers",
        """slow: marks tests as slow (deselect
 with '-m "not slow"')""",
    )
