import pytest
from pytest import approx
from asr.core import command, argument, option, returns, read_json,\
    describe, DictStr
from typing import List, NamedTuple


@command("test_recipe")
@argument("nx", help="Number of x's.")
@option("--ny", help="Optional number of y's.")
@returns("x", description="Description of x.", type=List[float])
@returns("y", description="Description of y.", type=List[float])
def test_recipe(nx: int, ny: int = 4):
    """Return a list of threes and fours."""
    x = [3] * nx
    y = [4] * ny
    return {'x': x, 'y': y}


class ASRResults:
    """Results base class."""

    pass


class GapResults(ASRResults):
    """Results from get_gap_info."""
    gap: dir


class Results(ASRResults):
    """Some results.

    gap: float
        The band gap.

    """

    gap: float
    gaps_nosoc: GapResults


# Issues: restarting already started calculation
# Locking reading of results file.

@command("test_recipe",
         dependencies=[test_recipe],
         version='1.0')
@argument("nx",
          type=int,
          description="Number of x's.",
          cli_type='option',
          cli_typecast=DictStr(),
          cli_help='CLI help for x')
@argument("ny", description="Optional number of y's.")
@argument("gs_filename",
          description="Filename for created file.",
          side_effect=True)
def test_recipe_dependency(nx: int,
                           ny: float = 4,
                           gs_filename: str = 'gs.gpw') -> Results:
    """Return a list of threes and fours.

    Returns
    -------
    dict

        x: List[int]
            List of x's

    """
    x = [3] * nx
    y = [4] * ny
    return {'x': x, 'y': y}


gsresults = gs()

gs.params.gs_filename


@command("test_recipe")
@argument("nx", help="Number of x's.")
@option("--ny", help="Optional number of y's.")
@returns("x", description="Description of x.", type=List[float])
@returns("y", description="Description of y.", type=List[float])
def test_recipe_returns_dict(nx: int, ny: int = 4) -> NamedTuple:
    """Return a list of threes and fours."""
    return {'something': {'a': 3}}


@pytest.fixture
def recipe():
    """Return a simple recipe."""
    return test_recipe


@pytest.mark.ci
def test_recipe_multiple_runs(asr_tmpdir, recipe):
    reciperesults3 = recipe(nx=3)
    assert reciperesults3["x"] == [3] * 3
    assert recipe.has_cache(nx=3)
    assert recipe.get_cache(nx=3) == reciperesults3
    reciperesults4 = recipe(nx=4)
    assert reciperesults4["x"] == [3] * 4
    assert recipe.has_cache(nx=3)
    assert recipe.has_cache(nx=4)
    assert recipe.get_cache(nx=3) == reciperesults3
    assert recipe.get_cache(nx=4) == reciperesults4


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
    assert reciperesults["x"] == [3] * 3
    assert reciperesults["y"] == [4] * 4

    assert reciperesults["__params__"]["nx"] == 3
    assert reciperesults["__params__"]["ny"] == 4

    assert reciperesults["__resources__"]["time"] == approx(0.1, abs=0.1)
    assert reciperesults["__resources__"]["ncores"] == 1
