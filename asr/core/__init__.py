import os
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Union, List

import click
import numpy as np
from ase.io import jsonio
from ase.parallel import parprint
import ase.parallel as parallel
import inspect
import copy
from ast import literal_eval
from functools import update_wrapper


def parse_dict_string(string, dct=None):
    if dct is None:
        dct = {}

    # Locate ellipsis
    string = string.replace('...', 'None:None')
    tmpdct = literal_eval(string)
    recursive_update(tmpdct, dct)
    return tmpdct


def recursive_update(dct, defaultdct):
    if None in dct:
        # This marks that we take default values from defaultdct
        del dct[None]
        for key in defaultdct:
            if key not in dct:
                dct[key] = defaultdct[key]

    for key, value in dct.items():
        if isinstance(value, dict) and None in value:
            if key not in defaultdct:
                del value[None]
                continue
            if not isinstance(defaultdct[key], dict):
                del value[None]
                continue
            recursive_update(dct[key], defaultdct[key])


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
    """Wrapper class for constructing recipes.

    This class implements the behaviour of an ASR recipe.

    This class wrappes a callable `func` and automatically endows the function
    with a command-line interface (CLI) through `cli` method. The CLI is
    defined using the :func:`asr.core.__init__.argument` and
    :func:`asr.core.__init__.option` functions in the core sub-package.

    The ASRCommand... XXX
    """

    package_dependencies = ('asr', 'ase', 'gpaw')

    def __init__(self, main,
                 module=None,
                 requires=None,
                 dependencies=None,
                 creates=None,
                 log=None,
                 webpanel=None,
                 save_results_file=True,
                 tests=None,
                 resources=None):
        """Construct an instance of an ASRCommand.

        Parameters
        ----------
        func : callable
            Wrapped function that

        """
        assert callable(main), 'The wrapped object should be callable'

        if module is None:
            module = main.__module__
            if module == '__main__':
                import inspect
                mod = inspect.getmodule(main)
                module = str(mod).split('\'')[1]

        name = f'{module}@{main.__name__}'

        # By default we omit @main if function is called main
        if name.endswith('@main'):
            name = name.replace('@main', '')

        # Function to be executed
        self._main = main
        self.name = name

        # Does the wrapped function want to save results files?
        self.save_results_file = save_results_file

        # What files are created?
        self._creates = creates

        # Is there additional information to log about the current execution
        self.log = log

        # Properties of this function
        self._requires = requires

        # Tell ASR how to present the data in a webpanel
        self.webpanel = webpanel

        # Commands can have dependencies. This is just a list of
        # pack.module.module@function that points to other functions
        # dot name like "recipe.name".
        self.dependencies = dependencies or []

        # Figure out the parameters for this function
        if not hasattr(self._main, '__asr_params__'):
            self._main.__asr_params__ = {}

        import copy
        self.myparams = copy.deepcopy(self._main.__asr_params__)

        import inspect
        sig = inspect.signature(self._main)
        self.signature = sig

        myparams = []
        defparams = {}
        for key, value in sig.parameters.items():
            assert key in self.myparams, \
                f'You havent provided a description for {key}'
            if value.default is not inspect.Parameter.empty:
                defparams[key] = value.default
            myparams.append(key)

        myparams = [k for k, v in sig.parameters.items()]
        for key in self.myparams:
            assert key in myparams, f'Param: {key} is unknown'
        self.defparams = defparams

        # Setup the CLI
        self.setup_cli()
        update_wrapper(self, self._main)

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
    def diskspace(self):
        if callable(self._diskspace):
            return self._diskspace()
        return self._diskspace

    @property
    def created_files(self):
        creates = []
        if self._creates:
            if callable(self._creates):
                creates += self._creates()
            else:
                creates += self._creates
        return creates

    @property
    def creates(self):
        creates = self.created_files
        if self.save_results_file:
            creates += [f'results-{self.name}.json']
        return creates

    @property
    def done(self):
        if Path(f'results-{self.name}.json').exists():
            return True
        return False

    def setup_cli(self):
        # Click CLI Interface
        CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

        cc = click.command
        co = click.option

        def clickify_docstring(doc):
            if doc is None:
                return
            doc_n = doc.split('\n')
            clickdoc = []
            skip = False
            for i, line in enumerate(doc_n):
                if skip:
                    skip = False
                    continue
                lspaces = len(line) - len(line.lstrip(' '))
                spaces = ' ' * lspaces
                bb = spaces + '\b'
                if line.endswith('::'):
                    skip = True

                    if not doc_n[i - 1].strip(' '):
                        clickdoc.pop(-1)
                        clickdoc.extend([bb, line, bb])
                    else:
                        clickdoc.extend([line, bb])
                elif ('-' in line
                      and (spaces + '-' * (len(line) - lspaces)) == line):
                    clickdoc.insert(-1, bb)
                    clickdoc.append(line)
                else:
                    clickdoc.append(line)
            doc = '\n'.join(clickdoc)

            return doc

        help = clickify_docstring(self._main.__doc__) or ''

        command = cc(context_settings=CONTEXT_SETTINGS,
                     help=help)(self.main)

        # Convert parameters into CLI Parameters!
        for name, param in self.myparams.items():
            param = param.copy()
            alias = param.pop('alias')
            argtype = param.pop('argtype')
            name2 = param.pop('name')
            assert name == name2
            assert name in self.myparams
            default = self.defparams.get(name, None)

            if argtype == 'option':
                command = co(show_default=True, default=default,
                             *alias, **param)(command)
            else:
                assert argtype == 'argument'
                command = click.argument(*alias, **param)(command)

        self._cli = command

    def cli(self, args=None):
        """Parse parameters from command line and call wrapped function.

        Parameters
        ----------
        args : List of strings or None
            List of command line arguments. If None: Read arguments from
            sys.argv.
        """
        return self._cli(standalone_mode=False,
                         prog_name=f'asr run {self.name}', args=args)

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)

    def main(self, *args, **kwargs):
        """Return results from wrapped function.

        This is the main function of an ASRCommand. It takes care of reading
        parameters, creating metadata, checksums etc. If you want to
        understand what happens when you execute an ASRCommand this is a good
        place to start.
        """
        # Run this recipes dependencies but only if it actually creates
        # a file that is in __requires__
        for dep in self.dependencies:
            recipe = get_recipe_from_name(dep)
            if recipe.done:
                continue

            filenames = set(self.requires).intersection(recipe.creates)
            if not all([Path(filename).exists() for filename in
                        filenames]):
                recipe()

        assert self.is_requirements_met(), \
            (f'{self.name}: Some required files are missing: {self.requires}. '
             'This could be caused by incorrect dependencies.')

        # Use the wrapped functions signature to create dictionary of
        # parameters
        bound_arguments = self.signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        params = dict(bound_arguments.arguments)
        for key, value in params.items():
            assert key in self.myparams, f'Unknown key: {key} {params}'
            # Default type
            if key not in self.defparams:
                continue
            if self.defparams[key] is None:
                continue
            tp = type(self.defparams[key])

            if tp != type(value):
                # Dicts has to be treated specially
                if tp == dict:
                    dct = copy.deepcopy(self.defparams[key])
                    dct = parse_dict_string(value, dct=dct)
                    params[key] = dct
                else:
                    params[key] = tp(value)

        # Read arguments from params.json and overwrite params
        if Path('params.json').is_file():
            paramsettings = read_json('params.json').get(self.name, {})
            for key, value in paramsettings.items():
                assert key in self.myparams, f'Unknown key: {key} {params}'
                params[key] = value

        paramstring = ', '.join([f'{key}={repr(value)}' for key, value in
                                 params.items()])
        parprint(f'Running {self.name}({paramstring})')

        tstart = time.time()
        # Execute the wrapped function
        with file_barrier(self.created_files, delete=False):
            results = self._main(**copy.deepcopy(params)) or {}
        results['__asr_name__'] = self.name
        tend = time.time()

        from ase.parallel import world
        results['__resources__'] = {'time': tend - tstart,
                                    'ncores': world.size}

        if self.log:
            log = results.get('__log__', {})
            log.update(self.log(**copy.deepcopy(params)))

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
        """Get parameter and software version information as a dictionary."""
        from ase.utils import search_current_git_hash
        exeinfo = {}
        modnames = self.package_dependencies
        versions = {}
        for modname in modnames:
            try:
                mod = import_module(modname)
            except ModuleNotFoundError:
                continue
            githash = search_current_git_hash(mod)
            version = mod.__version__
            if githash:
                versions[f'{modname}'] = f'{version}-{githash}'
            else:
                versions[f'{modname}'] = f'{version}'
        exeinfo['__versions__'] = versions

        return exeinfo


def command(*args, **kwargs):

    def decorator(func):
        return ASRCommand(func, *args, **kwargs)

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
    asrfolder = Path(__file__).parent.parent
    folders_with_recipes = [asrfolder / '.',
                            asrfolder / 'setup',
                            asrfolder / 'database']
    files = [filename for folder in folders_with_recipes
             for filename in folder.glob("[a-zA-Z]*.py")]
    modulenames = []
    for file in files:
        name = str(file.with_suffix(''))[len(str(asrfolder)):]
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
        # if not dependencies and hasattr(module, 'dependencies'):
        #     dependencies = module.dependencies

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


def encode_json(data):
    from ase.io.jsonio import MyEncoder
    return MyEncoder(indent=1).encode(data)


def write_json(filename, data):
    from pathlib import Path
    from ase.parallel import world

    with file_barrier([filename]):
        if world.rank == 0:
            Path(filename).write_text(encode_json(data))


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
def file_barrier(paths: List[Union[str, Path]], world=None,
                 delete=True):
    """Context manager for writing a file.

    After the with-block all cores will be able to read the file.

    Do "with file_barrier(['something.txt']):"

    This will remove the file, write the file and wait for the file.
    """
    if world is None:
        world = parallel.world

    for i, path in enumerate(paths):
        if isinstance(path, str):
            path = Path(path)
            paths[i] = path
        # Remove file:
        if delete:
            unlink(path, world)

    yield

    # Wait for file:
    i = 0
    world.barrier()
    while not all([path.is_file() for path in paths]):
        filenames = ', '.join([path.name for path in paths
                               if not path.is_file()])
        if i > 0:
            print(f'Waiting for ~{i}sec on existence of {filenames}'
                  ' on all ranks')
        time.sleep(1.0)
        i += 1
    world.barrier()


def singleprec_dict(dct):
    assert isinstance(dct, dict), f'Input {dct} is not dict.'

    for key, value in dct.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.int64:
                value = value.astype(np.int32)
            elif value.dtype == np.float64:
                value = value.astype(np.float32)
            elif value.dtype == np.complex128:
                value = value.astype(np.complex64)
            dct[key] = value
        elif isinstance(value, dict):
            dct[key] = singleprec_dict(value)

    return dct
