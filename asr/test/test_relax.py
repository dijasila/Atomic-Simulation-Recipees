from pathlib import Path
import contextlib
import pytest
import os


@contextlib.contextmanager
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


@pytest.mark.mpi(min_size=2)
def test_relax_cli_si(tmpdir):
    from asr.setup.materials import main as setupmaterial
    from asr.relax import main as relax

    with isolated_filesystem(tmpdir):
        setupmaterial.cli(["-s", "Si2"])
        Path("materials.json").rename("unrelaxed.json")
        relaxargs = (
            "{'mode':{'ecut':300,'dedecut':'estimate',...},"
            "'kpts':{'density':2,'gamma':True},...}"
        )
        relax.cli(["--calculator", relaxargs])


@pytest.mark.mpi
def test_size():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    assert comm.size > 0


@pytest.mark.mpi(min_size=2)
def test_size_2():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    assert comm.size >= 2
