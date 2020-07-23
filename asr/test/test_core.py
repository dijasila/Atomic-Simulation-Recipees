import pytest
from typing import Union
from pytest import approx
from ase import Atoms, Trajectory
from asr.core import command, argument, option, returns, read_json, DictStr, \
    AtomsStr, ASRCalculator, CalcStr, TrajectoryFile
from typing import List, NamedTuple
import numpy as np


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


# XXX What about created files.
# All objects must have a __hash__ attribute.


class ASRRelaxResults(ASRResults):
    """Results of relax recipe."""

    __results_version__ = 1
    structure: Atoms
    """My structure docs."""
    etot: float
    edft: float
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    spos: np.ndarray

    # TODO: Make descriptions a dict.
    KVP = ['a', 'b', 'c']
    descriptions = {'structure': 'The relaxed atoms object.',
                    'etot': ('Total energy of relaxed structure '
                             '(can include D3 correction).'),
                    'edft': ('Total energy of structure '
                             '(always excluding D3 correction).'),
                    'a_description': 'Cell parameter "a".',
                    'b_description': 'Cell parameter "b".',
                    'c_description': 'Cell parameter "c".',
                    'alpha_description': 'Cell parameter "alpha".',
                    'beta_description': 'Cell parameter "beta".',
                    'gamma_description': 'Cell parameter "gamma".',
                    'spos_description': 'Scaled atomic positions of relaxed structure.'}

    def webpanel(self):
        pass


@command('asr.relax',
         version=1,
         returns=ASRRelaxResults)
@argument("atoms",
          help='Atomic structure to be relaxed.',
          type=Atoms,
          cli_alias=['-a'],
          cli_argtype='option',
          cli_typecast=AtomsFromStr(),
          cli_default='unrelaxed.json')
@argument("outatoms",
          type=str,
          side_effect=True,
          cli_default='structure.json',
          cli_argtype='option',
          cli_typecast=str)
@argument("tmpatoms",
          type=Trajectory,
          side_effect=True,
          cli_argtype='option',
          cli_typecast=TrajectoryFromFile(must_exist=False),
          cli_default='relax.traj')
@argument("calculator",
          type=ASRCalculator,
          help='Calculator and its parameters.',
          has_package_dependencies=True,
          cli_typecast=CalcFromStr(),
          cli_argtype='option')
@argument('d3',
          type=bool,
          help='Relax with vdW D3.',
          cli_argtype='flag',
          cli_default=False)
@argument('fixcell',
          type=bool,
          help="Don't relax stresses.",
          cli_argtype='flag',
          cli_default=False)
@argument('allow-symmetry-breaking',
          type=bool,
          help='Allow symmetries to be broken during relaxation.',
          cli_argtype='flag',
          cli_default=False)
@argument('fmax',
          type=float,
          help='Force convergence criterium.',
          cli_argtype='option',
          cli_typecast=float,
          cli_default=1e-2)
@argument('enforce_symmetry',
          help='Symmetrize forces and stresses.',
          type=bool,
          cli_argtype='flag',
          cli_default=False)
def recipe(
        atoms: Atoms,
        outatoms: str = 'structure.json',
        calculator: ASRCalculator = ASRCalculator(
            {
                'name': 'gpaw',
                'mode': {'name': 'pw', 'ecut': 800},
                'xc': 'PBE',
                'kpts': {'density': 6.0, 'gamma': True},
                'basis': 'dzp',
                'symmetry': {'symmorphic': False},
                'convergence': {'forces': 1e-4},
                'txt': 'relax.txt',
                'occupations': {'name': 'fermi-dirac',
                                'width': 0.05},
                'charge': 0,
            }
        ),
        tmp_atoms: Union[Trajectory, None] = None,
        d3: bool = False,
        fixcell: bool = False,
        allow_symmetry_breaking: bool = False,
        fmax: float = 0.01,
        enforce_symmetry: bool = False) -> ASRRelaxResults:
    """Use all functionality recipe."""
    # ...
    # Do something and relax structure.
    # ...
    structure = Atoms()
    structure.write(outatoms)

    a, b, c = 3, 3, 3
    alpha, beta, gamma = 90, 90, 90
    etot = -100
    edft = -99

    # Raise Error if key is unknown or missing.
    return ASRRelaxResults(structure=structure,
                           etot=etot,
                           edft=edft,
                           a=a,
                           b=b,
                           c=c,
                           alpha=alpha,
                           beta=beta,
                           gamma=gamma,
                           spos=structure.get_scaled_positions())


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
