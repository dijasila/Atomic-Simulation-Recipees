import pytest
from pytest import approx


@pytest.mark.ci
def test_core(separate_folder):
    from click.testing import CliRunner
    from asr.core import command, argument, option, read_json
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
    result = runner.invoke(test_recipe._cli, ['--help'])
    assert result.exit_code == 0, result
    assert '-h, --help    Show this message and exit.' in result.output

    result = runner.invoke(test_recipe._cli, ['-h'])
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
