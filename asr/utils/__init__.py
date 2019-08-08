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


def add_param(func, param):
    if not hasattr(func, '__asr_params__'):
        func.__asr_params__ = {}

    name = param['name']
    assert name not in func.__asr_params__, \
        f'Double assignment of {name}'

    import inspect
    sig = inspect.signature(func)
    assert name in sig.parameters, f'Unkown parameter {name}'

    assert 'argtype' in param, \
        'You have to specify the parameter type: option or argument'

    if param['argtype'] == 'option':
        assert param['nargs'] == 1, 'Options only allow one argument'
    elif param['argtype'] == 'argument':
        assert param['default'] is None, 'Argument don\'t allow defaults'
    else:
        raise AssertionError(f'Unknown argument type {param["argtype"]}')

    func.__asr_params__[name] = param


def option(help=None, type=None, default=None, *args):

    def decorator(func):
        assert args, 'You have to give a name to this parameter'

        for arg in args:
            if arg.startswith('--'):
                name = arg[2:].replace('-', '_')
                break
        else:
            raise AssertionError('You must give exactly one alias that starts '
                                 'with -- and matches a function argument')
        param = {'argtype': 'option',
                 'help': help,
                 'type': type,
                 'default': default,
                 'alias': args,
                 'name': name,
                 'nargs': 1}

        add_param(func, param)

    return decorator


def argument(name, help=None, type=None, nargs=1, default=None):

    def decorator(func):
        assert not default, 'Arguments do not support defaults!'
        param = {'argtype': 'argument',
                 'help': help,
                 'type': type,
                 'default': default,
                 'alias': (name, ),
                 'name': name,
                 'nargs': nargs}

        add_param(func, param)

    return decorator


class Recipe:
    def __init__(self, main=None, webpanel=None):

        if main:
            assert isinstance(main, ASRCommand), \
                'main function is not ASR command'

        self.main = main
        self.webpanel = webpanel

    @classmethod
    def frompath(cls, name, reload=True):
        """Use like: Recipe.frompath('asr.relax')"""
        import importlib
        module = importlib.import_module(f'{name}')
        if reload:
            module = importlib.reload(module)

        main = None
        webpanel = None
        if hasattr(module, 'main'):
            main = module.main

        if hasattr(module, 'webpanel'):
            webpanel = module.webpanel

        return cls(main=main, webpanel=webpanel)


class ASRCommand:
    _asr_command = True

    def __init__(self, callback,
                 asr_name=None,
                 known_exceptions=None,
                 save_results_file=True,
                 add_skip_opt=True,
                 additional_callback='postprocessing',
                 tests=None,
                 resources=None,
                 creates=None):
        assert asr_name, 'You have to give a name to your ASR command!'
        assert callable(callback), 'The wrapped object should be callable'

        # Function to be executed
        self._callback = callback
        self.name = asr_name

        # We can handle these exceptions
        self.known_exceptions = known_exceptions or {}

        # Does the wrapped function want to save results files?
        self.asr_results_file = save_results_file

        # Do we want to add a skip deps option
        self.add_skip_opt = add_skip_opt

        # What files are created?
        self._creates = creates

        # What about resources?
        self._resources = resources

        # We can also have a postprocessing function to
        # run after the main function
        self.postprocessing = None

        # Our function can also have tests
        self.tests = tests

        # Figure out the parameters for this function
        if not hasattr(self._callback, '__asr_params__'):
            self._callback.__asr_params__ = {}

        self.params = self._callback.__asr_params__
        import inspect
        sig = inspect.signature(self._callback)
        self.signature = sig

        for key, value in sig.parameters.items():
            assert key in self.params, \
                f'You havent provided a description for {key}'
            assert value.default is inspect.Parameter.empty, \
                (f'Don\'t give default parameters in function definition. '
                 'Please use the @option and @argument ASR decorators')

        defparams = [k for k, v in sig.parameters.items()]
        for key in self.params:
            assert key in defparams, f'Param: {key} is unknown'

    def cli(self):
        CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
        func = self.callback
        for name, param in self.params.items():
            param = param.copy()
            alias = param.pop('alias')
            param.pop('name')
            func = click.option(*alias, **param)(func)

        command = click.command(callback=self.callback,
                                context_settings=CONTEXT_SETTINGS)
        return command(standalone_mode=False)

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)

    def main(self, skip_deps=False, catch_exceptions=True,
             *args, **kwargs):
        # Run all dependencies
        if not skip_deps:
            commands = get_dep_tree(self.name)
            for command in commands[:-1]:
                if not command.done():
                    command(skip_deps=False)

        # Try to run this command
        try:
            parprint(f'Running {self.name}')
            return self.callback(*args, **kwargs)
        except Exception as e:
            if type(e) in self.known_exceptions:
                parameters = self.known_exceptions[type(e)]
                # Update context
                if catch_exceptions:
                    parprint(f'Caught known exception: {type(e)}. '
                             'Trying again.')
                    for key in parameters:
                        kwargs[key] *= parameters[key]
                    return self.main(*args, **kwargs, catch_exceptions=False)
                else:
                    # We only allow the capture of one exception
                    parprint(f'Caught known exception: {type(e)}. '
                             'ERROR: I already caught one exception, '
                             'and I can at most catch one.')
                    raise

    def collect(self):
        import re
        kvp = {}
        key_descriptions = {}
        data = {}
        if self.done():
            name = self.name[4:]
            resultfile = f'results_{self._asr_name}.json'
            results = read_json(resultfile)
            if '__key_descriptions__' in results:
                tmpkd = {}
                for key, desc in key_descriptions.items():
                    descdict = {'type': None,
                                'iskvp': False,
                                'shortdesc': '',
                                'longdesc': '',
                                'units': ''}
                    if isinstance(desc, dict):
                        descdict.update(desc)
                        tmpkd[key] = desc
                        continue

                    assert isinstance(desc, str), \
                        'Key description has to be dict or str.'
                    # Get key type
                    desc, keytype = desc.split('->')
                    if keytype:
                        descdict['type'] = keytype

                    # Is this a kvp?
                    iskvp = '<KVP>' in desc
                    descdict['iskvp'] = iskvp
                    desc = desc.replace('<KVP>', '')

                    # Find units
                    m = re.search(r"\[(\w+)\]", desc)
                    unit = m.group(1) if m else ''
                    if unit:
                        descdict['units'] = unit
                    desc = desc.replace(unit, '')

                    # Find short description
                    m = re.search(r"\((\w+)\)", desc)
                    shortdesc = m.group(1) if m else ''

                    # The results is the long description
                    longdesc = desc.replace(shortdesc, '').strip()
                    if longdesc:
                        descdict['longdesc'] = longdesc
                    tmpkd[key] = descdict

                for key, desc in descdict.items():
                    key_descriptions[key] = \
                        (desc['shortdesc'], desc['longdesc'], desc['units'])

            key = f'results_{name}'
            msg = f'{self.name}: You cannot put a {key} in data'
            assert key not in data, msg
            data[key] = results

        return kvp, key_descriptions, data

    @property
    def creates(self):
        creates = []
        if self._creates:
            if callable(self._creates):
                creates += self._creates()
            else:
                creates += self._creates
        return creates

    def done(self):
        for file in self.creates:
            if not Path(file).exists():
                return False
        return True

    def callback(self, *args, **kwargs):
        # This is the actual function that is executed
        results = None

        # Skip main callback?
        if not self.done():
            # Figure out which parameters the function takes
            results = self._callback(*args, **kwargs)

        if results is None and self.postprocessing:
            # Then the results are calculated
            # by another callback function
            func = getattr(self.module, self.additional_callback)
            results = func()
            results['__params__'] = kwargs

        if self.asr_results_file:
            name = self._asr_name[4:]
            write_json(f'results_{name}.json', results)

            # Clean up possible tmpresults files
            tmppath = Path(f'tmpresults_{name}.json')
            if tmppath.exists():
                unlink(tmppath)

        return results


def command(name, add_skip_opt, *args, **kwargs):

    def decorator(func):
        return ASRCommand(func, name, *args, **kwargs)

    return decorator


def old_command(name, overwrite_params={},
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
    files = list(folder.glob('**/[a-zA-Z]*.py'))
    modulenames = []
    for file in files:
        if 'utils' in str(file) or 'tests' in str(file):
            continue
        name = str(file.with_suffix(''))[len(str(folder)):]
        modulename = 'asr' + name.replace('/', '.')
        modulenames.append(modulename)
    return modulenames


def get_recipes(sort=True, exclude=True, group=None):
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
        if not recipe.main.dependencies:
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
    return Recipe.frompath(name)


def get_dep_tree(name, reload=True):
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


if __name__ == '__main__':

    def function(a, b=2):
        return {'add': a + b}

    wrap = ASRCommand(function, 'function')

    print(wrap)
