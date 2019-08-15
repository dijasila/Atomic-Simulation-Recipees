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
import inspect
import copy


def md5sum(filename):
    from hashlib import md5
    hash = md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * hash.block_size), b""):
            hash.update(chunk)
    return hash.hexdigest()


def paramerrormsg(func, msg):
    return f'Problem in {func.__module__}@{func.__name__}. {msg}'


def add_param(func, param):
    if not hasattr(func, '__asr_params__'):
        func.__asr_params__ = {}

    name = param['name']
    assert name not in func.__asr_params__, \
        paramerrormsg(func, f'Double assignment of {name}')

    import inspect
    sig = inspect.signature(func)
    assert name in sig.parameters, \
        paramerrormsg(func, f'Unkown parameter {name}')

    assert 'argtype' in param, \
        paramerrormsg(func, 'You have to specify the parameter '
                      'type: option or argument')

    if param['argtype'] == 'option':
        if 'nargs' in param:
            assert param['nargs'] > 0, \
                paramerrormsg(func, 'Options only allow one argument')
    elif param['argtype'] == 'argument':
        assert 'default' not in param, \
            paramerrormsg(func, 'Argument don\'t allow defaults')
    else:
        raise AssertionError(
            paramerrormsg(func,
                          f'Unknown argument type {param["argtype"]}'))

    func.__asr_params__[name] = param


def option(*args, **kwargs):

    def decorator(func):
        assert args, 'You have to give a name to this parameter'

        for arg in args:
            params = inspect.signature(func).parameters
            name = arg.lstrip('-').split('/')[0].replace('-', '_')
            if name in params:
                break
        else:
            raise AssertionError(
                paramerrormsg(func,
                              'You must give exactly one alias that starts '
                              'with -- and matches a function argument.'))
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

    package_dependencies = ('asr', 'ase', 'gpaw')

    def __init__(self, main,
                 module=None,
                 requires=None,
                 dependencies=None,
                 creates=None,
                 tests=None,
                 resources='1:10m',
                 diskspace=0,
                 restart=0,
                 webpanel=None,
                 overwrite_defaults=None,
                 known_exceptions=None,
                 save_results_file=True,
                 pass_params=False,
                 add_skip_opt=True,
                 todict=None):
        assert callable(main), 'The wrapped object should be callable'

        if module is None:
            module = main.__module__

        name = f'{module}@{main.__name__}'

        # By default we omit @main if function is called main
        if name.endswith('@main'):
            name = name.replace('@main', '')

        # Function to be executed
        self._main = main
        self.name = name

        # We can handle these exceptions
        self.known_exceptions = known_exceptions or {}

        # Does the wrapped function want to save results files?
        self.save_results_file = save_results_file

        # Pass a dictionary with all params to the function for
        # convenience?
        self.pass_params = pass_params

        # What files are created?
        self._creates = creates

        # Properties of this function
        self._resources = resources
        self._diskspace = diskspace
        self._requires = requires
        self.restart = restart

        # Add skip dependencies option to control this?
        self.add_skip_opt = add_skip_opt

        # Tell ASR how to present the data in a webpanel
        self.webpanel = webpanel

        # Commands can have dependencies. This is just a list of
        # pack.module.module@function that points to other functions
        # dot name like "recipe.name".
        self.dependencies = dependencies or []

        # Our function can also have tests
        self.tests = tests

        # Figure out the parameters for this function
        if not hasattr(self._main, '__asr_params__'):
            self._main.__asr_params__ = {}

        self.todict = todict

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
            if value.default and \
               value.default is not inspect.Parameter.empty:
                defparams[key] = value.default
            myparams.append(key)

        myparams = [k for k, v in sig.parameters.items()]
        for key in self.params:
            assert key in myparams, f'Param: {key} is unknown'
        self.defparams = defparams

    @property
    def requires(self):
        if self._requires:
            if callable(self._requires):
                return self._requires()
            else:
                return self._requires
        return []

    def is_requirements_met(self):
        for filename in self.requires:
            if not Path(filename).is_file():
                return False
        return True

    @property
    def resources(self):
        if callable(self._resources):
            return self._resources()
        return self._resources

    @property
    def diskspace(self):
        if callable(self._diskspace):
            return self._diskspace()
        return self._diskspace

    @property
    def creates(self):
        creates = []
        if self.save_results_file:
            creates += [f'results-{self.name}.json']
        if self._creates:
            if callable(self._creates):
                creates += self._creates()
            else:
                creates += self._creates
        return creates

    @property
    def done(self):
        for file in self.creates:
            if not Path(file).exists():
                return False
        return True

    def cli(self):
        # Click CLI Interface
        CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

        cc = click.command
        co = click.option

        help = self._main.__doc__ or ''

        add_info = ''

        if self.save_results_file:
            add_info += ('Results stored in: '
                         f'results-{self.name}.json\n')

        if self.creates:
            add_info += f'Creates files: {self.creates}\n'

        if self.dependencies:
            add_info += f'Dependencies: {self.dependencies}\n'

        if self.resources:
            add_info += f'Resources: {self.resources}\n'

        if self.diskspace:
            add_info += f'Diskspace: {self.diskspace}\n'

        if self.diskspace:
            add_info += f'Diskspace: {self.diskspace}\n'

        if self.restart:
            add_info += f'Number restarts allowed: {self.restart}\n'

        if add_info:
            help += '\n\nASR metadata:\n\n\b\n' + add_info
            
        command = cc(context_settings=CONTEXT_SETTINGS,
                     help=help)(self.main)

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
                command = click.argument(*alias, **param)(command)
                
        if self.add_skip_opt:
            command = co('--skip-deps', is_flag=True, default=False,
                         help='Skip execution of dependencies')(command)

        return command(standalone_mode=False,
                       prog_name=f'asr run {self.name}')

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)

    def main(self, skip_deps=False, catch_exceptions=True,
             *args, **kwargs):

        if self.requires and self.is_requirements_met():
            # All requirements are met and we shouldn't run dependencies
            skip_deps = True

        if not skip_deps:
            deps = get_dep_tree(self.name)
            for name in deps[:-1]:
                recipe = get_recipe_from_name(name)
                for filename in recipe.creates:
                    if not Path(filename).is_file() and \
                       filename in self.requires:
                        recipe(skip_deps=True)

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

        assert self.is_requirements_met(), \
            f'Some required files are missing: {self.requires}'

        # Use the wrapped functions signature to create dictionary of
        # parameters
        params = copy.deepcopy(self.defparams)
        params.update(dict(self.signature.bind(*args, **kwargs).arguments))

        # Read arguments from params.json if not already in params
        paramsettings = {}
        if Path('params.json').is_file():
            paramsettings = read_json('params.json').get(self.name, {})
            for key, value in paramsettings.items():
                assert key in self.params, f'Unknown key: {key} {params}'

                # If any parameters have been given directly to the function
                # we don't use the ones from the param.json file
                params[key] = value

        parprint(f'Running {self.name}')

        # Execute the wrapped function
        if self.pass_params:
            results = self._main(params, **params) or {}
        else:
            results = self._main(**params) or {}

        # Do we have to store some digests of previous calculations?
        if self.creates:
            results['__creates__'] = {}
            for filename in self.creates:
                if filename.startswith('results-'):
                    # Don't log own results file
                    continue
                hexdigest = md5sum(filename)
                results['__creates__'][filename] = hexdigest

        # Also make hexdigests of results-files for dependencies
        if self.requires:
            results['__requires__'] = {}
            for filename in self.requires:
                hexdigest = md5sum(filename)
                results['__requires__'][filename] = hexdigest

        # Save parameters
        if params:
            results.update({'__params__': params})

        # Update with hashes for packages dependencies
        results.update(self.get_execution_info())

        if self.save_results_file:
            name = self.name
            write_json(f'results-{name}.json', results)

            # Clean up possible tmpresults files
            tmppath = Path(f'tmpresults-{name}.json')
            if tmppath.exists():
                unlink(tmppath)

        return results

    def get_execution_info(self):
        """Get parameter and software version information as a dictionary"""
        from ase.utils import search_current_git_hash
        exeinfo = {}
        modnames = self.package_dependencies
        versions = {}
        for modname in modnames:
            mod = import_module(modname)
            githash = search_current_git_hash(mod)
            version = mod.__version__
            if githash:
                versions[f'{modname}'] = f'{version}-{githash}'
            else:
                versions[f'{modname}'] = f'{version}'
        exeinfo['__versions__'] = versions

        return exeinfo

    def collect(self):
        import re
        kvp = {}
        key_descriptions = {}
        data = {}

        resultfile = f'results-{self.name}.json'
        results = read_json(resultfile)
        msg = f'{self.name}: You cannot put a {resultfile} in data'
        assert resultfile not in data, msg
        data[resultfile] = results

        extra_files = results.get('__requires__', {})
        extra_files.update(results.get('__creates__', {}))
        if extra_files:
            for filename, checksum in extra_files.items():
                if filename in data:
                    continue
                file = Path(filename)
                if not file.is_file():
                    print(f'Warning: Required file {filename}'
                          ' doesn\'t exist')

                filetype = file.suffix
                if filetype == '.json':
                    dct = read_json(filename)
                elif self.todict and filetype in self.todict:
                    dct = self.todict[filetype](filename)
                    dct['__md5__'] = md5sum(filename)
                else:
                    dct = {'pointer': str(file.absolute()),
                           '__md5__': md5sum(filename)}
                data[filename] = dct

        # Parse key descriptions to get long,
        # short, units and key value pairs
        if '__key_descriptions__' in results:
            tmpkd = {}

            for key, desc in results['__key_descriptions__'].items():
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
                desc, *keytype = desc.split('->')
                if keytype:
                    descdict['type'] = keytype

                # Is this a kvp?
                iskvp = desc.startswith('KVP:')
                descdict['iskvp'] = iskvp
                desc = desc.replace('KVP:', '').strip()

                # Find units
                m = re.search(r"\[(\w+)\]", desc)
                unit = m.group(1) if m else ''
                if unit:
                    descdict['units'] = unit
                desc = desc.replace(f'[{unit}]', '').strip()

                # Find short description
                m = re.search(r"\((\w+)\)", desc)
                shortdesc = m.group(1) if m else ''

                # The results is the long description
                longdesc = desc.replace(f'({shortdesc})', '').strip()
                if longdesc:
                    descdict['longdesc'] = longdesc
                tmpkd[key] = descdict

            for key, desc in tmpkd.items():
                key_descriptions[key] = \
                    (desc['shortdesc'], desc['longdesc'], desc['units'])

                if key in results and desc['iskvp']:
                    kvp[key] = results[key]

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

    def __call__(self, params, *args, **kwargs):
        # Try to read sub-result from previous calculation
        subresult = self.read_subresult()
        if subresult is None:
            subresult = self.calculate(params, *args, **kwargs)

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

    def calculate(self, params, *args, **kwargs):
        """Do the actual calculation"""
        subresult = self.calculator.__call__(*args, **kwargs)
        assert isinstance(subresult, dict)

        subresult.update(self.get_execution_info(params))
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


def get_recipe_module_names():
    # Find all modules containing recipes
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
    # Get the tree of dependencies from recipe of "name"
    # by following dependencies of dependencies
    import importlib

    tmpdeplist = [name]

    for i in range(1000):
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
            tmpdeplist.append(dependency)
    else:
        raise AssertionError('Unreasonably many dependencies')

    tmpdeplist.reverse()
    deplist = []
    for dep in tmpdeplist:
        if dep not in deplist:
            deplist.append(dep)

    return deplist


def get_recipe_modules():
    # Get recipe modules
    import importlib
    modules = get_recipe_module_names()

    mods = []
    for module in modules:
        mod = importlib.import_module(module)
        mods.append(mod)
    return mods


def get_recipes():
    # Get all recipes in all modules
    modules = get_recipe_modules()

    functions = []
    for module in modules:
        for attr in module.__dict__:
            attr = getattr(module, attr)
            if isinstance(attr, ASRCommand):
                functions.append(attr)
    return functions


def get_recipe_from_name(name):
    # Get a recipe from a name like asr.gs@postprocessing
    import importlib
    mod, func = parse_mod_func(name)
    module = importlib.import_module(mod)
    return getattr(module, func)


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
    from ase.io import read
    atoms = read('structure.json')
    nd = int(np.sum(atoms.get_pbc()))
    return nd


mag_elements = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'}


def magnetic_atoms(atoms):
    return np.array([symbol in mag_elements
                     for symbol in atoms.get_chemical_symbols()],
                    dtype=bool)


def write_json(filename, data):
    from pathlib import Path
    from ase.io.jsonio import MyEncoder
    from ase.parallel import world

    with file_barrier(filename):
        if world.rank == 0:
            Path(filename).write_text(MyEncoder(indent=1).encode(data))


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

    world.barrier()
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
