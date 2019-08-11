import os
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Union

import click
import numpy as np
from ase.io import jsonio
from ase.parallel import parprint
import ase.parallel as parallel


def md5sum(filename):
    from hashlib import md5
    hash = md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * hash.block_size), b""):
            hash.update(chunk)
    return hash.hexdigest()


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
        assert 'nargs' not in param, 'Options only allow one argument'
    elif param['argtype'] == 'argument':
        assert 'default' not in param, 'Argument don\'t allow defaults'
    else:
        raise AssertionError(f'Unknown argument type {param["argtype"]}')

    func.__asr_params__[name] = param


def option(*args, **kwargs):

    def decorator(func):
        assert args, 'You have to give a name to this parameter'

        for arg in args:
            if arg.startswith('--'):
                name = arg[2:].split('/')[0].replace('-', '_')
                break
        else:
            raise AssertionError('You must give exactly one alias that starts '
                                 'with -- and matches a function argument')
        param = {'argtype': 'option',
                 'alias': args,
                 'name': name}
        param.update(kwargs)
        add_param(func, param)
        return func

    return decorator


def argument(name, **kwargs):

    def decorator(func):
        assert 'default' not in kwargs, 'Arguments do not support defaults!'
        param = {'argtype': 'argument',
                 'alias': (name, ),
                 'name': name}
        param.update(kwargs)
        add_param(func, param)
        return func
        
    return decorator


class ASRCommand:

    def __init__(self, main,
                 module=None,
                 overwrite_defaults=None,
                 known_exceptions=None,
                 save_results_file=True,
                 add_skip_opt=True,
                 creates=None,
                 dependencies=None,
                 resources='1:10m',
                 diskspace=0,
                 tests=None,
                 restart=0):
        assert callable(main), 'The wrapped object should be callable'

        if module is None:
            module = main.__module__

        name = f'{module}@{main.__name__}'

        # Function to be executed
        self._main = main
        self.name = name

        # We can handle these exceptions
        self.known_exceptions = known_exceptions or {}

        # Does the wrapped function want to save results files?
        self.save_results_file = save_results_file

        # What files are created?
        self._creates = creates

        # Properties of this function
        self._resources = resources
        self._diskspace = diskspace
        self.restart = restart

        # Add skip dependencies option to control this?
        self.add_skip_opt = add_skip_opt

        # Commands can have dependencies. This is just a list of
        # pack.module.module@function that points to other function.
        # If no @function then we assume function=main
        self.dependencies = []
        if dependencies:
            for dep in dependencies:
                if '@' not in dep:
                    dep = dep + '@main'
                self.dependencies.append(dep)

        # Our function can also have tests
        self.tests = tests

        # Figure out the parameters for this function
        if not hasattr(self._main, '__asr_params__'):
            self._main.__asr_params__ = {}

        import copy
        self.params = copy.deepcopy(self._main.__asr_params__)

        import inspect
        sig = inspect.signature(self._main)
        self.signature = sig

        myparams = []
        defparams = {}
        for key, value in sig.parameters.items():
            assert key in self.params, \
                f'You havent provided a description for {key}'
            if value.default:
                defparams[key] = value.default
            myparams.append(key)

        myparams = [k for k, v in sig.parameters.items()]
        for key in self.params:
            assert key in myparams, f'Param: {key} is unknown'
        self.defparams = defparams

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
        creates = []
        creates += self.creates
        if self.save_results_file:
            creates += ['results-{self.name}.json']

        for file in creates:
            if not Path(file).exists():
                return False
        return True

    def cli(self):
        # Click CLI Interface
        CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

        cc = click.command
        co = click.option
        command = cc(context_settings=CONTEXT_SETTINGS)(self.main)

        # Convert out parameters into CLI Parameters!
        for name, param in self.params.items():
            param = param.copy()
            alias = param.pop('alias')
            argtype = param.pop('argtype')
            name2 = param.pop('name')
            assert name == name2
            assert name in self.params
            default = self.defparams.get(name, None)

            if argtype == 'option':
                command = co(show_default=True, default=default,
                             *alias, **param)(command)
            else:
                assert argtype == 'argument'
                command = click.argument(show_default=True,
                                         *alias, **param)(command)

        if self.add_skip_opt:
            command = co('--skip-deps', is_flag=True, default=False,
                         help='Skip execution of dependencies')(command)

        return command(standalone_mode=False)

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)

    def main(self, skip_deps=False, catch_exceptions=True,
             *args, **kwargs):
        # Run all dependencies
        if not skip_deps:
            deps = get_dep_tree(self.name)
            for name in deps[:-1]:
                function = get_function_from_name(name)
                if not function.done():
                    function(skip_deps=True)
                else:
                    print(f'Dependency {name} already done!')
        # Try to run this command
        try:
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

    def callback(self, *args, **kwargs):
        # This is the main function of an ASRCommand. It takes care of
        # reading parameters can creating metadata, checksums.
        # If you to understand what happens when you execute an ASRCommand
        # this is a good place to start

        print(f'Running {self.name}')

        # Use the wrapped functions signature to create dictionary of
        # parameters
        params = dict(self.signature.bind(*args, **kwargs).arguments)

        # Read arguments from params.json
        if Path('params.json').is_file():
            paramsettings = read_json('params.json')
            pardefaults = paramsettings.get(self.name, {})
            for key, value in pardefaults.items():
                assert key in self.params, f'Unknown key: {key} {params}'

                # If any parameters have been given directly to the function
                # we don't use the ones from the param.json file
                if key not in params:
                    params[key] = value

        # Execute the wrapped function
        results = self._main(**params)

        if not results:
            results = {}

        results['__md5_digest__'] = {}
        for filename in self.creates:
            hexdigest = md5sum(filename)
            results['__md5_digest__'][filename] = hexdigest

        # Also make hexdigests of resuls-files for dependencies
        for dep in self.dependencies:
            filename = f'results-{dep}.json'
            hexdigest = md5sum(filename)
            results['__md5_digest__'][dep] = hexdigest

        results.update(get_execution_info(params))

        if self.save_results_file:
            name = self.name
            write_json(f'results-{name}.json', results)

            # Clean up possible tmpresults files
            tmppath = Path(f'tmpresults-{name}.json')
            if tmppath.exists():
                unlink(tmppath)

        return results

    def collect(self):
        import re
        kvp = {}
        key_descriptions = {}
        data = {}
        if self.done():
            name = self.name[4:]
            resultfile = f'results_{self.name}.json'
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


def command(*args, **kwargs):

    def decorator(func):
        return ASRCommand(func, *args, **kwargs)

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


def parse_mod_func(name):
    # Split a module function reference like
    # asr.relax@main into asr.relax and main.
    mod, *func = name.split('@')
    if not func:
        func = ['main']

    assert len(func) == 1, \
        'You cannot have multiple : in your function description'

    return mod, func[0]


def get_dep_tree(name, reload=True):
    import importlib

    tmpdeplist = ['@'.join(parse_mod_func(name))]

    for i in range(100):
        if i == len(tmpdeplist):
            break
        dep = tmpdeplist[i]
        mod, func = parse_mod_func(dep)
        module = importlib.import_module(mod)

        assert hasattr(module, func), f'{module}.{func} doesn\'t exist'
        function = getattr(module, func)
        dependencies = function.dependencies
        if not dependencies and hasattr(module, 'dependencies'):
            dependencies = module.dependencies

        for dependency in dependencies:
            depname = '@'.join(parse_mod_func(dependency))
            tmpdeplist.append(depname)

    tmpdeplist.reverse()
    deplist = []
    for dep in tmpdeplist:
        if dep not in deplist:
            deplist.append(dep)

    return deplist


def get_function_from_name(name):
    import importlib
    mod, func = parse_mod_func(name)
    module = importlib.import_module(mod)
    return getattr(module, func)


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

    def function(a, b, c, d, e=5, f=4):
        return {'add': a + b}

    # wrap = ASRCommand(function, 'function')

    import inspect

    args = (2,)
    sig = inspect.signature(function)
    print(dict(sig.bind(2, 3, 4, 5).arguments))

    # print(wrap())
