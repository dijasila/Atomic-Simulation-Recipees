import pytest
from pytest import approx
from asr.core import command, argument, option, ASRResult, Parameters
from time import sleep
from asr.core.root import ASRRootNotFound


@command("test_recipe")
@argument("nx")
@option("--ny", help="Optional number of y's")
def tmp_recipe(nx, ny=4) -> ASRResult:
    x = [3] * nx
    y = [4] * ny
    return {'x': x, 'y': y}


@pytest.fixture
def recipe():
    """Return a simple recipe."""
    return tmp_recipe


@pytest.mark.ci
def test_recipe_defaults(asr_tmpdir, recipe):
    """Test that recipe.get_defaults returns correct defaults."""
    defaults = recipe.defaults
    assert defaults == Parameters({'ny': 4})


@pytest.mark.ci
def test_recipe_setting_new_defaults(asr_tmpdir, recipe):
    """Test that defaults set in params.json are correctly applied."""
    from asr.core import write_json
    params = {'test_recipe:tmp_recipe': {'ny': 5}}
    write_json('params.json', params)
    defaults = recipe.defaults
    assert defaults == Parameters({'ny': 5})


@pytest.mark.ci
def test_recipe_setting_overriding_defaults(asr_tmpdir, recipe):
    """Test that defaults are correctly overridden when setting parameter."""
    record = recipe.get(3, 3)
    assert record.parameters == Parameters({'nx': 3, 'ny': 3})
    assert record.result['x'] == [3] * 3
    assert record.result['y'] == [4] * 3


@command("asr.test.test_core")
@argument("nx")
@option("--ny", help="Optional number of y's")
def a_recipe(nx, ny=4) -> ASRResult:
    x = [3] * nx
    y = [4] * ny
    sleep(0.1)
    return {'x': x, 'y': y}


@pytest.mark.ci
def test_core(asr_tmpdir_w_params):
    """Test some simple properties of a recipe."""
    from click.testing import CliRunner
    from asr.core.command import make_cli_command

    runner = CliRunner()
    cmd = make_cli_command(a_recipe)
    result = runner.invoke(cmd, ['--help'])
    assert result.exit_code == 0, result
    assert '-h, --help    Show this message and exit.' in result.output

    result = runner.invoke(cmd, ['-h'])
    assert result.exit_code == 0
    assert '-h, --help    Show this message and exit.' in result.output

    a_recipe(nx=3)
    record = a_recipe.get(nx=3)

    assert record

    assert record.result["x"] == [3] * 3
    assert record.result["y"] == [4] * 4

    assert record.parameters["nx"] == 3
    assert record.parameters["ny"] == 4

    assert record.resources.execution_duration == approx(0.1, abs=0.1)
    assert record.resources.ncores == 1


@pytest.mark.ci
def test_not_initialized(recipe, tmp_path):
    """Test fail behaviour when running in uninitialized directory."""
    from ase.utils import workdir

    with workdir(tmp_path):
        with pytest.raises(ASRRootNotFound,
                           match='Root directory not initialized'):
            recipe(3.0)


@pytest.mark.ci
def test_recipe_has():
    from ase import Atoms
    from asr.convex_hull import main
    assert not main.has(atoms=Atoms(), databases=['does_not_exist'])
