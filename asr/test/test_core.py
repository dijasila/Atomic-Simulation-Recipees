import pytest
from pytest import approx
from asr.core import command, argument, option, read_json


@pytest.fixture
def recipe():
    """Return a simple recipe."""

    @command("test_recipe")
    @argument("nx")
    @option("--ny", help="Optional number of y's")
    def test_recipe(nx, ny=4):
        x = [3] * nx
        y = [4] * ny
        return {'x': x, 'y': y}

    return test_recipe


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
    assert results['__params__'] == {'nx': 3, 'ny': 3}
    assert results['x'] == [3] * 3
    assert results['y'] == [4] * 3


@pytest.mark.ci
def test_core(asr_tmpdir_w_params):
    """Test some simple properties of a recipe."""
    from click.testing import CliRunner
    from pathlib import Path
    from time import sleep

    @command("test_recipe")
    @argument("nx")
    @option("--ny", help="Optional number of y's")
    def test_recipe(nx, ny=4):
        x = [3] * nx
        y = [4] * ny
        sleep(0.1)
        return {'x': x, 'y': y}

    runner = CliRunner()
    result = runner.invoke(test_recipe.setup_cli(), ['--help'])
    assert result.exit_code == 0, result
    assert '-h, --help    Show this message and exit.' in result.output

    result = runner.invoke(test_recipe.setup_cli(), ['-h'])
    assert result.exit_code == 0
    assert '-h, --help    Show this message and exit.' in result.output

    test_recipe(nx=3)

    resultfile = Path('results-test_recipe@test_recipe.json')
    assert resultfile.is_file()

    reciperesults = read_json(resultfile)
    assert all(reciperesults["x"] == [3] * 3)
    assert all(reciperesults["y"] == [4] * 4)

    assert reciperesults["__params__"]["nx"] == 3
    assert reciperesults["__params__"]["ny"] == 4

    assert reciperesults["__resources__"]["time"] == approx(0.1, abs=0.1)
    assert reciperesults["__resources__"]["ncores"] == 1
