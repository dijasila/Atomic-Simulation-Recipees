import os
import time
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Union, List

import click
from ase.io import jsonio
from ase.parallel import parprint
import ase.parallel as parallel
import inspect
import copy
from ast import literal_eval


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


def get_recipe_name_from_function(function):
    import inspect
    file = Path(inspect.getfile(function))
    folder = file.absolute().parent
    # Figure out if this is a module
    initfile = folder / '__init__.py'
    folders = []
    if initfile.is_file():
        # Then this is a package
        while (folder / '__init__.py').is_file():
            folders.append(folder)
            folder = folder.parent
    folders.reverse()
    if folders:
        module = '.'.join([folder.name for folder in folders])
        return f'{module}.{file.name}@{function.__name__}'
    else:
        return '{file.name}@{function.__name__}'


class ASRCommand:

    package_dependencies = ('asr', 'ase', 'gpaw')

    def __init__(self, callback,
                 requires=None,
                 dependencies=None,
                 creates=None,
                 tests=None,
                 webpanel=None,
                 add_skip_opt=True):
        assert callable(callback), 'The wrapped object should be callable'

        # Function to be executed
        self.callback = callback
        self.name = get_recipe_name_from_function(callback)

        # What files are created?
        self._creates = creates

        # What files are required to run?
        self._requires = requires

        # Tell ASR how to present the data in a webpanel
        self.webpanel = webpanel

        # Commands can have dependencies. This is just a list of
        # pack.module.module@function that points to other functions
        # dot name like "recipe.name".
        self.dependencies = dependencies or []

        # Our function can also have tests
        self.tests = tests

        # Figure out the parameters for this function
        if not hasattr(self.callback, '__asr_params__'):
            self.callback.__asr_params__ = {}

        import copy
        self.myparams = copy.deepcopy(self.callback.__asr_params__)

        import inspect
        sig = inspect.signature(self.callback)
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

        self.__doc__ = self.callback.__doc__

    @property
    def state(self):
        """The state of tests of this recipe.
        Currently only supports 'tested' and 'untested'"""
        if not self.tests:
            return 'untested'
        return 'tested'

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
                elif ('-' in line and
                      (spaces + '-' * (len(line) - lspaces)) == line):
                    clickdoc.insert(-1, bb)
                    clickdoc.append(line)
                else:
                    clickdoc.append(line)
            doc = '\n'.join(clickdoc)

            return doc

        help = clickify_docstring(self.callback.__doc__) or ''

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
                
        if self.add_skip_opt:
            command = co('--skip-deps', is_flag=True, default=False,
                         help='Skip execution of dependencies.')(command)

        self._cli = command

    def cli(self, *args, **kwargs):
        return self._cli(standalone_mode=False,
                         prog_name=f'asr run {self.name}', *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)

    def main(self, skip_deps=False, *args, **kwargs):

        if not skip_deps:
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

        # Try to run this command
        results = self.callback(*args, **kwargs)

        return results

    def _main(self, *args, **kwargs):
        # This is the main function of an ASRCommand. It takes care of
        # reading parameters can creating metadata, checksums.
        # If you to understand what happens when you execute an ASRCommand
        # this is a good place to start

        assert self.is_requirements_met(), \
            (f'Some required files are missing: {self.requires}. '
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

        # Execute the wrapped function
        with file_barrier(self.created_files, delete=False):
            results = self.callback(**copy.deepcopy(params)) or {}
        results['__asr_name__'] = self.name

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

        name = self.name
        write_json(f'results-{name}.json', results)

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


def write_json(filename, data):
    from pathlib import Path
    from ase.io.jsonio import MyEncoder
    from ase.parallel import world

    with file_barrier([filename]):
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
def file_barrier(paths: List[Union[str, Path]], world=None,
                 delete=True):
    """Context manager for writing a file.

    After the with-block all cores will be able to read the file.

    >>> with file_barrier(['something.txt']):
    ...     <write file>
    ...

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
    while not all([path.is_file() for path in paths]):
        time.sleep(1.0)
    world.barrier()
