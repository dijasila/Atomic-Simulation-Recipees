import os
from contextlib import contextmanager
from functools import partial
import click
import numpy as np
from importlib import import_module
from ase.io import jsonio
from ase.parallel import parprint
option = partial(click.option, show_default=True)
argument = click.argument


class ASRCommand(click.Command):
    _asr_command = True

    def __init__(self, asr_name=None, known_exceptions=None,
                 save_results_file=True,
                 *args, **kwargs):
        assert asr_name, 'You have to give a name to your ASR command!'
        self._asr_name = asr_name
        self.known_exceptions = known_exceptions or {}
        self.asr_results_file = save_results_file
        click.Command.__init__(self, *args, **kwargs)

    def main(self, *args, **kwargs):
        return click.Command.main(self, standalone_mode=False,
                                  *args, **kwargs)

    def invoke(self, ctx):
        # Pop the skip deps argument
        skip_deps = ctx.params.pop('skip_deps')
        self.invoke_wrapped(ctx, skip_deps)

    def invoke_wrapped(self, ctx, skip_deps=False, catch_exceptions=True):
        if skip_deps:
            args = ['--skip-deps']
        else:
            args = []

        # Run all dependencies
        recipes = get_dep_tree(self._asr_name)
        for recipe in recipes[:-1]:  # Don't include itself
            if not recipe.done():
                recipe.main(args=args)

        try:
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
                    # We only allow the capture of one exception
                    return self.invoke_wrapped(ctx, catch_exceptions=False)
                else:
                    parprint(f'Caught known exception: {type(e)}. '
                             'Already caught one exception, '
                             'and I can at most catch one.')
            raise
        if not results:
            results = {}
        results['__params__'] = ctx.params
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
        results['__versions__'] = versions

        if self.asr_results_file:
            name = self._asr_name[4:]
            write_json(f'results-{name}.json', results)
        return results


def command(name, overwrite_params={}, *args, **kwargs):
    params = get_parameters(name)
    params.update(overwrite_params)

    ud = update_defaults

    CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

    def decorator(func):

        cc = click.command(cls=ASRCommand,
                           context_settings=CONTEXT_SETTINGS,
                           asr_name=name,
                           *args, **kwargs)

        func = option('--skip-deps/--run-deps', is_flag=True, default=False,
                      help="Don't skip dependencies?")(func)
        
        if hasattr(func, '__click_params__'):
            func = cc(ud(name, params)(func))
        else:
            func = cc(func)

        return func
    
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
excludelist = ['asr.gw', 'asr.hse', 'asr.piezoelectrictensor',
               'asr.bse', 'asr.gapsummary']


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
            sortedrecipes.append(recipe)

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


def get_dep_tree(name):
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
    recipes = [Recipe.frompath(names[ind]) for ind in indices]
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


def read_json(filename):
    from pathlib import Path
    dct = jsonio.decode(Path(filename).read_text())
    return dct
