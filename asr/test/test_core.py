import pytest
from pytest import approx
from asr.core import command, argument, option, read_json, ASRResult
from time import sleep


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
    defaults = recipe.get_defaults()
    assert defaults == {'ny': 4}


@pytest.mark.ci
def test_recipe_setting_new_defaults(asr_tmpdir, recipe):
    """Test that defaults set in params.json are correctly applied."""
    from asr.core import write_json
    params = {'test_recipe@test_recipe': {'ny': 5}}
    write_json('params.json', params)
    defaults = recipe.get_defaults()
    assert defaults == {'ny': 5}


@pytest.mark.ci
def test_recipe_setting_overriding_defaults(asr_tmpdir, recipe):
    """Test that defaults are correctly overridden when setting parameter."""
    results = recipe(3, 3)
    assert results.metadata.params == {'nx': 3, 'ny': 3}
    assert results['x'] == [3] * 3
    assert results['y'] == [4] * 3


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
    from pathlib import Path

    runner = CliRunner()
    result = runner.invoke(a_recipe.setup_cli(), ['--help'])
    assert result.exit_code == 0, result
    assert '-h, --help    Show this message and exit.' in result.output

    result = runner.invoke(a_recipe.setup_cli(), ['-h'])
    assert result.exit_code == 0
    assert '-h, --help    Show this message and exit.' in result.output

    a_recipe(nx=3)

    resultfile = Path('results-asr.test.test_core@a_recipe.json')
    assert resultfile.is_file()

    reciperesults = read_json(resultfile)
    assert reciperesults["x"] == [3] * 3
    assert reciperesults["y"] == [4] * 4

    assert reciperesults.metadata.params["nx"] == 3
    assert reciperesults.metadata.params["ny"] == 4

    assert reciperesults.metadata.resources["time"] == approx(0.1, abs=0.1)
    assert reciperesults.metadata.resources["ncores"] == 1
