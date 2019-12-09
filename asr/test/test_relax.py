from pathlib import Path
import contextlib
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


def test_relax_cli_si(tmpdir):
    from asr.setup.materials import main as setupmaterial
    from asr.relax import main as relax
    from asr.core import read_json

    with isolated_filesystem(tmpdir):
        setupmaterial.cli(["-s", "Si2"])
        Path("materials.json").rename("unrelaxed.json")
        relaxargs = (
            "{'mode':{'ecut':200,'dedecut':'estimate',...},"
            "'kpts':{'density':1,'gamma':True},...}"
        )
        relax.cli(["--calculator", relaxargs])
        results = read_json("results-asr.relax.json")
        assert abs(results["c"] - 3.978) < 0.001


def test_relax_cli_bn(tmpdir):
    from asr.setup.materials import main as setupmaterial
    from asr.relax import main as relax
    from asr.core import read_json

    with isolated_filesystem(tmpdir):
        setupmaterial.cli(["-s", "BN,natoms=2"])
        Path("materials.json").rename("unrelaxed.json")
        relaxargs = (
            "{'mode':{'ecut':300,'dedecut':'estimate',...},"
            "'kpts':{'density':2,'gamma':True},...}"
        )
        relax.cli(["--calculator", relaxargs])

        results = read_json("results-asr.relax.json")
        assert results["c"] > 5
