import os
import time
from contextlib import contextmanager
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Union

import click
import numpy as np
from ase.io import jsonio
from ase.parallel import parprint
import ase.parallel as parallel


option = partial(click.option, show_default=True)
argument = click.argument


class ASRCommand(click.Command):
    _asr_command = True

    def __init__(self, asr_name=None, known_exceptions=None,
                 save_results_file=True,
                 add_skip_opt=True, callback=None,
                 additional_callback='postprocessing',
                 tests=None,
                 creates=None,
                 dependencies=None,
                 diskspace=None,
                 restart=None,
                 group=None,
                 resources=None,
                 postprocessing=None,
                 collect=None,
                 webpanel=None,
                 *args, **kwargs):

        self.known_exceptions = known_exceptions or {}
        self.asr_results_file = save_results_file
        self.add_skip_opt = add_skip_opt
        self._callback = callback

        self.module = import_module(asr_name)

        # Metadata for our ASR recipes
        assert asr_name, 'You have to give a name to your ASR command!'
        self._asr_name = asr_name
        self.creates = creates or self._creates
        self.dependencies = dependencies or self._dependencies
        self.diskspace = diskspace or self._diskspace
        self.restart = restart or self._restart
        # self.group = group or self._group
        self.resources = resources or self._resources
        self.tests = tests or self._tests

        # Extra functionality for our ASR functionality
        self.postprocessing = postprocessing or self._postprocessing
        self.collect = collect or self._collect
        self.webpanel = webpanel or self._webpanel

        click.Command.__init__(self, callback=self.callback, *args, **kwargs)

    @property
    def _creates(self):
        creates = [f'results_{self._asr_name}.json']
        if hasattr(self.module, 'creates'):
            creates += self.module.creates
        return creates

    @property
    def _dependencies(self):
        dependencies = []
        if hasattr(self.module, 'dependencies'):
            dependencies += self.module.dependencies
        return dependencies

    @property
    def _resources(self):
        resources = '1:10m'
        if hasattr(self.module, 'resources'):
            resources = self.module.resources
        return resources

    @property
    def _diskspace(self):
        diskspace = 0
        if hasattr(self.module, 'diskspace'):
            diskspace = self.module.diskspace
        return diskspace

    @property
    def _restart(self):
        restart = 0
        if hasattr(self.module, 'restkart'):
            restart = self.module.restart
        return restart

    def main(self, *args, **kwargs):
        return click.Command.main(self, standalone_mode=False,
                                  *args, **kwargs)

    def postprocessing(self):
        pass

    def webpanel(self):
        pass

    def collect(self):
        kvp = {}
        key_descriptions = {}
        data = {}
        if self.done():
            if self.module.collect_data:
                kvp, key_descriptions, data = self.collect_data(atoms)

            name = self.name[4:]
            resultfile = Path(f'results_{name}.json')
            from ase.io import jsonio
            results = jsonio.decode(resultfile.read_text())
            key = f'results_{name}'
            msg = f'{self.name}: You cannot put a {key} in data'
            assert key not in data, msg
            data[key] = results

        return kvp, key_descriptions, data

    def done(self):
        if self.creates:
            for file in self.creates:
                if not Path(file).exists():
                    return False
            return True
        return False

    def invoke(self, ctx):
        """Invoke a recipe.

        By default, invoking a recipe also means invoking its dependencies.
        This can be avoided using the --skip-deps keyword."""
        # Pop the skip deps argument
        if self.add_skip_opt:
            skip_deps = ctx.params.pop('skip_deps')
        else:
            skip_deps = True

        # Run all dependencies
        if not skip_deps:
            recipes = get_dep_tree(self._asr_name)
            for recipe in recipes[:-1]:
                if not recipe.done():
                    recipe.run(args=['--skip-deps'])

        return self.invoke_myself(ctx)

    def invoke_myself(self, ctx, catch_exceptions=True):
        """Invoke my own callback."""
        try:
            parprint(f'Running {self._asr_name}')
            results = click.Command.invoke(self, ctx)
        except Exception as e:
            if type(e) in self.known_exceptions:
                parameters = self.known_exceptions[type(e)]
                # Update context
                if catch_exceptions:
                    parprint(f'Caught known exception: {type(e)}. '
                             'Trying again.')
                    for key in parameters:
                        ctx.params[key] *= parameters[key]
                    return self.invoke_myself(ctx, catch_exceptions=False)
                else:
                    # We only allow the capture of one exception
                    parprint(f'Caught known exception: {type(e)}. '
                             'ERROR: I already caught one exception, '
                             'and I can at most catch one.')
            raise

        if not results:
            results = {}

        results.update(get_execution_info(ctx.params))
        
        if self.asr_results_file:
            name = self._asr_name[4:]
            write_json(f'results_{name}.json', results)

            # Clean up possible tmpresults files
            tmppath = Path(f'tmpresults_{name}.json')
            if tmppath.exists():
                unlink(tmppath)

        return results

    def callback(self, *args, **kwargs):
        results = None
        # Skip main callback?
        if not self.done():
            results = self._callback(*args, **kwargs)

        if results is None and hasattr(self.module, self.additional_callback):
            # Then the results are calculated by another callback function
            func = getattr(self.module, self.additional_callback)
            results = func()
        return results


def command(name, overwrite_params={},
            add_skip_opt=True, *args, **kwargs):
    params = get_parameters(name)
    params.update(overwrite_params)

    ud = update_defaults

    CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

    def decorator(func):

        cc = click.command(cls=ASRCommand,
                           context_settings=CONTEXT_SETTINGS,
                           asr_name=name,
                           add_skip_opt=add_skip_opt,
                           *args, **kwargs)

        if add_skip_opt:
            func = option('--skip-deps/--run-deps', is_flag=True,
                          default=False,
                          help="Skip execution of dependencies?")(func)

        if hasattr(func, '__click_params__'):
            func = cc(ud(name, params)(func))
        else:
            func = cc(func)

        return func

    return decorator


class ASRSubResult:
    def __init__(self, asr_name, calculator):
        self._asr_name = asr_name[4:]
        self.calculator = calculator
        self._asr_key = calculator.__name__

        self.results = {}

    def __call__(self, ctx, *args, **kwargs):
        # Try to read sub-result from previous calculation
        subresult = self.read_subresult()
        if subresult is None:
            subresult = self.calculate(ctx, *args, **kwargs)

        return subresult

    def read_subresult(self):
        """Read sub-result from tmpresults file if possible"""
        subresult = None
        path = Path(f'tmpresults_{self._asr_name}.json')
        if path.exists():
            self.results = jsonio.decode(path.read_text())
            # Get subcommand sub-result, if available
            if self._asr_key in self.results.keys():
                subresult = self.results[self._asr_key]

        return subresult

    def calculate(self, ctx, *args, **kwargs):
        """Do the actual calculation"""
        subresult = self.calculator.__call__(*args, **kwargs)
        assert isinstance(subresult, dict)

        subresult.update(get_execution_info(ctx.params))
        self.results[self._asr_key] = subresult

        write_json(f'tmpresults_{self._asr_name}.json', self.results)

        return subresult


def subresult(name):
    """Decorator pattern for sub-result"""
    def decorator(calculator):
        return ASRSubResult(name, calculator)

    return decorator


@contextmanager
def chdir(folder, create=False, empty=False):
    dir = os.getcwd()
    if empty and folder.is_dir():
        import shutil
        shutil.rmtree(str(folder))
    if create and not folder.is_dir():
        os.mkdir(folder)
    os.chdir(str(folder))
    yield
    os.chdir(dir)


# We need to reduce this list to only contain collect
excludelist = ['asr.gapsummary']


def get_execution_info(params):
    """Get parameter and software version information as a dictionary"""
    exceinfo = {'__params__': params}

    from ase.utils import search_current_git_hash
    modnames = ['asr', 'ase', 'gpaw']
    versions = {}
    for modname in modnames:
        mod = import_module(modname)
        githash = search_current_git_hash(mod)
        version = mod.__version__
        if githash:
            versions[f'{modname}'] = f'{version}-{githash}'
        else:
            versions[f'{modname}'] = f'{version}'
    exceinfo['__versions__'] = versions

    return exceinfo


def get_all_recipe_names():
    from pathlib import Path
    folder = Path(__file__).parent.parent
    files = list(folder.glob('[a-zA-Z]*.py'))
    files += list(folder.glob('setup/[a-zA-Z]*.py'))
    modulenames = []
    for file in files:
        name = str(file.with_suffix(''))[len(str(folder)):]
        modulename = 'asr' + name.replace('/', '.')
        modulenames.append(modulename)
    return modulenames


def get_recipes(sort=True, exclude=True, group=None):
    from asr.utils.recipe import Recipe
    names = get_all_recipe_names()
    recipes = []
    for modulename in names:
        if modulename in excludelist:
            continue
        recipe = Recipe.frompath(modulename)
        if group and not recipe.group == group:
            continue
        recipes.append(recipe)

    if sort:
        recipes = sort_recipes(recipes)

    return recipes


def sort_recipes(recipes):
    sortedrecipes = []

    # Add the recipes with no dependencies (these must exist)
    for recipe in recipes:
        if not recipe.dependencies:
            if recipe.group in ['postprocessing', 'property']:
                sortedrecipes.append(recipe)
            else:
                sortedrecipes = [recipe] + sortedrecipes

    assert len(sortedrecipes), 'No recipes without deps!'
    for i in range(1000):
        for recipe in recipes:
            names = [recipe.name for recipe in sortedrecipes]
            if recipe.name in names:
                continue
            for dep in recipe.dependencies:
                if dep not in names:
                    break
            else:
                sortedrecipes.append(recipe)

        if len(recipes) == len(sortedrecipes):
            break
    else:
        names = [recipe.name for recipe in recipes]
        msg = ('Something went wrong when parsing dependencies! '
               f'Input recipes: {names}')
        raise AssertionError(msg)
    return sortedrecipes


def get_recipe(name):
    from asr.utils import Recipe
    return Recipe.frompath(name)


def get_dep_tree(name, reload=True):
    from asr.utils.recipe import Recipe
    names = get_all_recipe_names()
    indices = [names.index(name)]
    for j in range(100):
        if not indices[j:]:
            break
        for ind in indices[j:]:
            recipe = Recipe.frompath(names[ind])
            if not hasattr(recipe, 'dependencies'):
                continue
            deps = recipe.dependencies
            if not deps:
                continue
            for dep in deps:
                index = names.index(dep)
                if index not in indices:
                    indices.append(index)
    else:
        raise RuntimeError('Dependencies are weird!')
    recipes = [Recipe.frompath(names[ind], reload=reload) for ind in indices]
    recipes = sort_recipes(recipes)
    return recipes


def get_parameters(key):
    from pathlib import Path
    if Path('params.json').is_file():
        params = read_json('params.json')
    else:
        params = {}

    params = params.get(key, {})
    return params


def is_magnetic():
    import numpy as np
    from ase.io import read
    atoms = read('structure.json')
    magmom_a = atoms.get_initial_magnetic_moments()
    maxmom = np.max(np.abs(magmom_a))
    if maxmom > 1e-3:
        return True
    else:
        return False


def get_dimensionality():
    import numpy as np
    from ase.io import read
    atoms = read('structure.json')
    nd = int(np.sum(atoms.get_pbc()))
    return nd


mag_elements = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'}


def magnetic_atoms(atoms):
    import numpy as np
    return np.array([symbol in mag_elements
                     for symbol in atoms.get_chemical_symbols()],
                    dtype=bool)


def update_defaults(key, params={}):
    params.update(get_parameters(key))

    def update_defaults_dec(func):
        fparams = func.__click_params__
        for externaldefault in params:
            for param in fparams:
                if externaldefault == param.name:
                    param.default = params[param.name]
                    break
            else:
                msg = f'{key}: {externaldefault} is unknown'
                raise AssertionError(msg)
        return func
    return update_defaults_dec


def get_start_parameters():
    import json
    with open('structure.json', 'r') as fd:
        asejsondb = json.load(fd)
    params = asejsondb.get('1').get('calculator_parameters', {})

    return params


def get_reduced_formula(formula, stoichiometry=False):
    """
    Returns the reduced formula corresponding to a chemical formula,
    in the same order as the original formula
    E.g. Cu2S4 -> CuS2

    Parameters:
        formula (str)
        stoichiometry (bool): if True, return the stoichiometry ignoring the
          elements appearing in the formula, so for example "AB2" rather than
          "MoS2"
    Returns:
        A string containing the reduced formula
    """
    from functools import reduce
    from fractions import gcd
    import string
    import re
    split = re.findall('[A-Z][^A-Z]*', formula)
    matches = [re.match('([^0-9]*)([0-9]+)', x)
               for x in split]
    numbers = [int(x.group(2)) if x else 1 for x in matches]
    symbols = [matches[i].group(1) if matches[i] else split[i]
               for i in range(len(matches))]
    divisor = reduce(gcd, numbers)
    result = ''
    numbers = [x // divisor for x in numbers]
    numbers = [str(x) if x != 1 else '' for x in numbers]
    if stoichiometry:
        numbers = sorted(numbers)
        symbols = string.ascii_uppercase
    for symbol, number in zip(symbols, numbers):
        result += symbol + number
    return result


def has_inversion(atoms, use_spglib=True):
    """
    Parameters:
        atoms: Atoms object
            atoms
        use_spglib: bool
            use spglib
    Returns:
        out: bool
    """
    try:
        import spglib
    except ImportError as x:
        import warnings
        warnings.warn('using gpaw symmetry for inversion instead: {}'
                      .format(x))
        use_spglib = False

    atoms2 = atoms.copy()
    atoms2.pbc[:] = True
    atoms2.center(axis=2)
    if use_spglib:
        R = -np.identity(3, dtype=int)
        r_n = spglib.get_symmetry(atoms2, symprec=1.0e-3)['rotations']
        return np.any([np.all(r == R) for r in r_n])
    else:
        from gpaw.symmetry import atoms2symmetry
        return atoms2symmetry(atoms2).has_inversion


def write_json(filename, data):
    from pathlib import Path
    from ase.io.jsonio import MyEncoder
    from ase.parallel import world
    if world.rank == 0:
        Path(filename).write_text(MyEncoder(indent=4).encode(data))
    world.barrier()


def read_json(filename):
    from pathlib import Path
    dct = jsonio.decode(Path(filename).read_text())
    return dct


def unlink(path: Union[str, Path], world=None):
    """Safely unlink path (delete file or symbolic link)."""

    if isinstance(path, str):
        path = Path(path)
    if world is None:
        world = parallel.world

    # Remove file:
    if world.rank == 0:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    else:
        while path.is_file():
            time.sleep(1.0)
    world.barrier()


@contextmanager
def file_barrier(path: Union[str, Path], world=None):
    """Context manager for writing a file.

    After the with-block all cores will be able to read the file.

    >>> with file_barrier('something.txt'):
    ...     <write file>
    ...

    This will remove the file, write the file and wait for the file.
    """

    if isinstance(path, str):
        path = Path(path)
    if world is None:
        world = parallel.world

    # Remove file:
    unlink(path, world)

    yield

    # Wait for file:
    while not path.is_file():
        time.sleep(1.0)
    world.barrier()
