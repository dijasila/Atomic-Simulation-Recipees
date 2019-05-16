import os
from contextlib import contextmanager
from functools import partial
import click
import numpy as np
option = partial(click.option, show_default=True)
argument = click.argument


class ASRCommand(click.Command):
    _asr_command = True

    def __call__(self, *args, **kwargs):
        return self.main(standalone_mode=False, *args, **kwargs)


def command(name, overwrite={}, *args, **kwargs):
    params = get_parameters(name)
    params.update(overwrite)

    ud = update_defaults

    CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

    def decorator(func):
        cc = click.command(cls=ASRCommand,
                           context_settings=CONTEXT_SETTINGS,
                           *args, **kwargs)
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


def get_recipes(sort=True, exclude=True):
    import importlib
    from pathlib import Path

    files = Path(__file__).parent.parent.glob('[a-zA-Z]*.py')
    recipes = []
    for file in files:
        name = file.with_suffix('').name
        modulename = f'asr.{name}'
        if modulename in excludelist:
            continue
        module = importlib.import_module(f'asr.{name}')
        recipes.append(module)

    if sort:
        sortedrecipes = []

        # Add the recipes with no dependencies (these must exist)
        for recipe in recipes:
            if not hasattr(recipe, 'dependencies'):
                sortedrecipes.append(recipe)
            else:
                if len(recipe.dependencies) == 0:
                    sortedrecipes.append(recipe)

        for i in range(1000):
            for recipe in recipes:
                names = [recipe.__name__ for recipe in sortedrecipes]
                if recipe.__name__ in names:
                    continue
                for dep in recipe.dependencies:
                    if dep not in names:
                        break
                else:
                    sortedrecipes.append(recipe)

            if len(recipes) == len(sortedrecipes):
                break
        else:
            msg = 'Something went wrong when parsing dependencies!'
            raise AssertionError(msg)
        recipes = sortedrecipes

    return recipes


def get_dep_tree(name):
    recipes = get_recipes(sort=True)

    names = [recipe.__name__ for recipe in recipes]
    indices = [names.index(name)]
    for j in range(100):
        if not indices[j:]:
            break
        for ind in indices[j:]:
            if not hasattr(recipes[ind], 'dependencies'):
                continue
            deps = recipes[ind].dependencies
            for dep in deps:
                index = names.index(dep)
                if index not in indices:
                    indices.append(index)
    else:
        raise RuntimeError('Dependencies are weird!')
    print(indices)
    indices = sorted(indices)
    return [recipes[ind] for ind in indices]


def get_parameters(key=None):
    from pathlib import Path
    import json
    if Path('params.json').is_file():
        with open('params.json', 'r') as fd:
            params = json.load(fd)
    else:
        params = {}

    if key and key in params:
        params = params[key]

    return params


def is_magnetic():
    import numpy as np
    atoms = get_start_atoms()
    magmom_a = atoms.get_initial_magnetic_moments()
    maxmom = np.max(np.abs(magmom_a))
    if maxmom > 1e-3:
        return True
    else:
        return False


def get_dimensionality():
    import numpy as np
    start = get_start_atoms()
    nd = int(np.sum(start.get_pbc()))
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
        for param in fparams:
            for externaldefault in params:
                if externaldefault == param.name:
                    param.default = params[param.name]
                    break

        return func
    return update_defaults_dec


def get_start_file():
    "Get starting atomic structure"
    from pathlib import Path
    fnames = list(Path('.').glob('start.*'))
    assert len(fnames) == 1, fnames
    return str(fnames[0])


def get_start_atoms():
    from ase.io import read
    fname = get_start_file()
    atoms = read(str(fname))
    return atoms


def get_start_parameters(atomfile=None):
    import json
    if atomfile is None:
        try:
            atomfile = get_start_file()
        except AssertionError:
            return {}
    with open(atomfile, 'r') as fd:
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
