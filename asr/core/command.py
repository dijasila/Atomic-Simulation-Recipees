"""Implement ASRCommand class and related decorators."""
from . import (read_json, write_file, md5sum,
               file_barrier, clickify_docstring, ASRResult,
               get_recipe_from_name)
import functools
from .temporary_directory import temporary_directory
from .dependencies import dependency_stack
from .cache import ASRCache
from typing import List, Dict
from ase.parallel import parprint
import atexit
import click
import copy
import time
from importlib import import_module
from pathlib import Path
import inspect
from functools import update_wrapper


def to_json(obj):
    """Write an object to a json file."""
    json_string = obj.format_as('json')
    return json_string


def get_md5_checksums(filenames: List[str]) -> Dict[str, str]:
    """Get md5 checksums of a list of files."""
    checksums = {}
    for filename in filenames:
        hexdigest = md5sum(filename)
        checksums[filename] = hexdigest
    return checksums


def does_files_exist(filenames: List[str]) -> List[bool]:
    """Check whether files exist."""
    return [Path(filename).is_file() for filename in filenames]


def format_param_string(params: dict):
    """Represent params as comma separated string."""
    return ', '.join([f'{key}={repr(value)}' for key, value in
                      params.items()])


def _paramerrormsg(func, msg):
    return f'Problem in {func.__module__}@{func.__name__}. {msg}'


def _add_param(func, param):
    if not hasattr(func, '__asr_params__'):
        func.__asr_params__ = {}

    name = param['name']
    assert name not in func.__asr_params__, \
        _paramerrormsg(func, f'Double assignment of {name}')

    import inspect
    sig = inspect.signature(func)
    assert name in sig.parameters, \
        _paramerrormsg(func, f'Unkown parameter {name}')

    assert 'argtype' in param, \
        _paramerrormsg(func, 'You have to specify the parameter '
                       'type: option or argument')

    if param['argtype'] == 'option':
        if 'nargs' in param:
            assert param['nargs'] > 0, \
                _paramerrormsg(func, 'Options only allow one argument')
    elif param['argtype'] == 'argument':
        assert 'default' not in param, \
            _paramerrormsg(func, 'Argument don\'t allow defaults')
    else:
        raise AssertionError(
            _paramerrormsg(func,
                           f'Unknown argument type {param["argtype"]}'))

    func.__asr_params__[name] = param


def option(*args, **kwargs):
    """Tag a function to have an option."""

    def decorator(func):
        assert args, 'You have to give a name to this parameter'

        for arg in args:
            params = inspect.signature(func).parameters
            name = arg.lstrip('-').split('/')[0].replace('-', '_')
            if name in params:
                break
        else:
            raise AssertionError(
                _paramerrormsg(func,
                               'You must give exactly one alias that starts '
                               'with -- and matches a function argument.'))
        param = {'argtype': 'option',
                 'alias': args,
                 'name': name}
        param.update(kwargs)
        _add_param(func, param)
        return func

    return decorator


def argument(name, **kwargs):
    """Mark a function to have an argument."""

    def decorator(func):
        assert 'default' not in kwargs, 'Arguments do not support defaults!'
        param = {'argtype': 'argument',
                 'alias': (name, ),
                 'name': name}
        param.update(kwargs)
        _add_param(func, param)
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
                 namespace=None,
                 dependencies=None,
                 creates=None,
                 returns=None,
                 log=None,
                 webpanel=None,
                 save_results_file=True,
                 tests=None,
                 resources=None,
                 version=None,
                 package_dependencies=None):
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
        name = f'{namespace}@{main.__name__}'

        # By default we omit @main if function is called main
        if name.endswith('@main'):
            name = name.replace('@main', '')

        # Function to be executed
        self._main = main
        self.name = name

        # Return type
        if returns is None:
            returns = ASRResult
        # assert returns is not None, 'Please specify a return type!'
        self.returns = returns

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
        self.cache = ASRCache('results-{name}.json')
        self.dependencies = dependencies or []

        # Figure out the parameters for this function
        if not hasattr(self._main, '__asr_params__'):
            self._main.__asr_params__ = {}

        import copy
        self.myparams = copy.deepcopy(self._main.__asr_params__)

        import inspect
        sig = inspect.signature(self._main)
        self.__signature__ = sig

        # Setup the CLI
        functools.update_wrapper(self, self._main)

    def get_signature(self):
        """Return signature with updated defaults based on params.json."""
        myparams = []
        for key, value in self.__signature__.parameters.items():
            assert key in self.myparams, \
                f'Missing description for param={key}.'
            myparams.append(key)

        # Check that all annotated parameters can be found in the
        # actual function signature.
        myparams = [k for k, v in self.__signature__.parameters.items()]
        for key in self.myparams:
            assert key in myparams, f'param={key} is unknown.'

        if Path('params.json').is_file():
            # Read defaults from params.json.
            paramsettings = read_json('params.json').get(self.name, {})
            if paramsettings:
                signature_parameters = dict(self.__signature__.parameters)
                for key, new_default in paramsettings.items():
                    assert key in signature_parameters, \
                        f'Unknown param in params.json: param={key}.'
                    parameter = signature_parameters[key]
                    signature_parameters[key] = parameter.replace(
                        default=new_default)

                new_signature = self.__signature__.replace(
                    parameters=[val for val in signature_parameters.values()])
                return new_signature

        return self.__signature__

    def get_defaults(self):
        """Get default parameters based on signature and params.json."""
        signature = self.get_signature()
        defparams = {}
        for key, value in signature.parameters.items():
            if value.default is not inspect.Parameter.empty:
                defparams[key] = value.default
        return defparams

    @property
    def diskspace(self):
        if callable(self._diskspace):
            return self._diskspace()
        return self._diskspace

    def get_parameters(self):
        """Get the parameters of this function."""
        return self.myparams

    def setup_cli(self):
        # Click CLI Interface
        CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

        cc = click.command
        co = click.option

        help = clickify_docstring(self._main.__doc__) or ''

        command = cc(context_settings=CONTEXT_SETTINGS,
                     help=help)(self.main)

        # Convert parameters into CLI Parameters!
        defparams = self.get_defaults()
        for name, param in self.get_parameters().items():
            param = param.copy()
            alias = param.pop('alias')
            argtype = param.pop('argtype')
            name2 = param.pop('name')
            assert name == name2
            assert name in self.myparams
            if 'default' in param:
                default = param.pop('default')
            else:
                default = defparams.get(name, None)

            if argtype == 'option':
                command = co(show_default=True, default=default,
                             *alias, **param)(command)
            else:
                assert argtype == 'argument'
                command = click.argument(*alias, **param)(command)

        return command

    def cli(self, args=None):
        """Parse parameters from command line and call wrapped function.

        Parameters
        ----------
        args : List of strings or None
            List of command line arguments. If None: Read arguments from
            sys.argv.
        """
        command = self.setup_cli()
        return command(standalone_mode=False,
                       prog_name=f'asr run {self.name}', args=args)

    def get_wrapped_function(self):
        """Return wrapped function."""
        return self._main

    def __call__(self, *args, **kwargs):
        """Delegate to self.main."""
        return self.main(*args, **kwargs)

    def apply_defaults(self, *args, **kwargs):
        """Apply defaults to args and kwargs.

        Reads the signature of the wrapped function and applies the
        defaults where relevant.

        """
        signature = self.get_signature()
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        params = copy.deepcopy(dict(bound_arguments.arguments))
        return params

    def main(self, *args, **kwargs):
        """Return results from wrapped function.

        This is the main function of an ASRCommand. It takes care of
        reading parameters, creating metadata, checksums etc. If you
        want to understand what happens when you execute an ASRCommand
        this is a good place to start.

        Implementation goes as follows::

            1. Parse input parameters
            2. Check if a cached result already exists
               and return that if it does.
            --- Otherwise
            3. Run all dependencies.
            4. Get execution metadata, ie., code_versions, created files and
               required files.

        """
        # Inspired by: lab-notebook, provenance, invoke, fabric, joblib
        # TODO: Tag results with random run #ID.
        # TODO: Converting old result files to new format.
        # TODO: Save date and time.
        # TODO: We should call external files side effects.
        # TODO: When to register current run as a dependency.
        # TODO: Locking reading of results file.
        # TODO: Use time stamp for hashing files as well.
        # TODO: Easy to design a system for pure functions,
        # but we need side effects as well.
        # TODO: Should we have an ignore keyword?
        # TODO: Some parameters need to know about others in order
        # to properly initialize, eg., in GPAW the poisson solver need
        # to know about the dimensionality to set dipole layer and also to
        # get the setup fingerprints, also 1D materials need higher
        # kpoint density.
        # TODO: SHA256 vs MD5 speeeeeeed?
        # TODO: All arguments requires a JSON serialization method.
        # TODO: How do Django make data migrations?
        # TODO: Require completely flat ASRResults data structure?
        # TODO: Should we have a way to Signal ASR (think click.Context)?
        # TODO: The caching database could be of a non-relational format (would be similar to current format).
        # TODO: Should Result objects have some sort of verification mechanism? Like checking acoustic sum rules?
        # TODO: Make clean run environment class?
        # TODO: Make an old type results object.

        # REQ: Recipe must be able to run multiple times and cache their results (think LRU-cache).
        # REQ: Must be possible to change implementation of recipe
        #      without invalidating previous results
        # REQ: Must support side-effects such as files written to disk.
        # REQ: Must store information about code versions
        # REQ: Must be able to read defaults from configuration file on
        #      a per-folder basis.
        # REQ: Must support chaining of recipes (dependencies).
        # REQ: Caching database should be simple and decentralized (think sqlite).
        # REQ: Caching database should be plain text.
        # REQ: Returned object should be self-contained (think ase BandStructure object).
        # REQ: Returned objects must be able to present themselves as figures and HTML.
        # REQ: Must be delocalized from ASR (ie. must be able to have a seperate set of recipes locally, non-related to asr).
        # REQ: Must also have a packaging mechanism for entire projects (ie. ASE databases).
        # REQ: Must be possible to call as a simple python function thus circumventing CLI.
        # REQ: Must be able to call without scripting, eg. through a CLI.
        # REQ: Must support all ASE calculators.

        parameters = Parameters(self.apply_defaults_to_parameters(*args, **kwargs))

        cached_result_info = self.cache.get_cached_result_info(
            parameters=parameters
        )

        if cached_result_info.is_execution_completed():
            result = cached_result_info.get_result()
            result.verify_side_effects()
            dependency_stack.register_result_id(result.id)
            parprint('Returning cached result for '
                     f'{self.name}({parameters})')
            return result

        code_versions = self.get_code_versions(parameters=parameters)
        temporary_files = self.get_temporary_files(parameters=parameters)

        if cached_result_info.is_initiated():
            result.validate(code_versions=code_versions,
                            version=self.version)
        else:
            assert not any(does_files_exist(temporary_files)), \
                ('Some temporary files already exist '
                 f'(temporary_files={temporary_files})')

        params = dict(bound_arguments.arguments)
        execution_directory = get_temporary_directory_name(result.id)
        self.cache.initiate(
            parameters,
            versions=code_versions,
            version=self.version,
            execution_directory=execution_directory,
        )

        parprint(f'Running {self.name}({parameter_string})')

        # We register an exit handler to handle unexpected exits.
        atexit.register(clean_files, files=temporary_files)

        if not isinstance(result, self.returns):
            assert isinstance(result, dict)
            result = self.returns(data=result)
        # Execute the wrapped function
        # Register dependencies implement stack like data structure.
        with dependency_stack as my_dependencies:
            for dependency in self.dependencies:
                dependency()

            with CleanEnvironment(temporary_directory) as env, \
                 clean_files(temporary_files), \
                 file_barrier(created_files, delete=False):
                tstart = time.time()
                results = self._main(**parameters)
                tend = time.time()

        from ase.parallel import world
        metadata = dict(asr_name=self.name,
                        resources=dict(time=tend - tstart,
                                       ncores=world.size,
                                       tstart=tstart,
                                       tend=tend),
                        params=params,
                        code_versions=get_execution_info(
                            self.package_dependencies))
        # Do we have to store some digests of previous calculations?
        if self.creates:
            metadata['creates'] = {}
            for filename in self.creates:
                if filename.startswith('results-'):
                    # Don't log own results file
                    continue
                hexdigest = md5sum(filename)
                metadata['creates'][filename] = hexdigest

        if self.requires:
            metadata['requires'] = {}
            for filename in self.requires:
                hexdigest = md5sum(filename)
                metadata['requires'][filename] = hexdigest

        result.metadata = metadata
        if self.save_results_file:
            name = self.name
            json_string = to_json(result)
            write_file(f'results-{name}.json', json_string)

        return result

        metadata = {
            'asr_name': self.name,
            'resources': {'time': tend - tstart,
                          'ncores': world.size},
        }
        created_md5_checksums = get_md5_checksums(created_files)
        results.update(results=results,
                       checksums=created_md5_checksums,
                       metadata=metadata)
        if self.save_cache:
            self.cache.add(cached_result)

        return cached_result




def get_execution_info(package_dependencies):
    """Get parameter and software version information as a dictionary."""
    from ase.utils import search_current_git_hash
    versions = {}
    for modname in package_dependencies:
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
    return versions


def command(*args, **kwargs):

    def decorator(func):
        return ASRCommand(func, *args, **kwargs)

    return decorator


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
            if isinstance(attr, ASRCommand) or hasattr(attr, 'is_recipe'):
                functions.append(attr)
    return functions
