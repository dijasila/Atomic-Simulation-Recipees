"""Implement ASRCommand class and related decorators."""
from . import (read_json, write_file, md5sum,
               file_barrier, clickify_docstring, ASRResult)
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


class Cache():

    def __init__(self):
        pass

    def add(self, result):
        pass

    def has(self, result):
        pass

    def get(self, result):
        pass


class Parameters:

    pass


class SimpleRunner():

    def __init__(self):
        pass

    def __call__(self, func, parameters):
        return self.run(func, parameters)

    def run(self, func, parameters):
        return func(**parameters)


class RunSpecification:

    def __init__(self, function: callable,
                 parameters: Parameters,
                 name: str,
                 codes: CodeDescriptor):
        self.parameters = parameters
        self.name = name
        self.function = function
        self.codes = codes


class RunRecord:

    def __init__(self):
        pass

def construct_workdir(run_specification: RunSpecification):
    pass


def register_dependencies(run_specification: RunSpecification):
    pass


def register_metadata(run_specification: RunSpecification):
    pass


def register_sideffects(run_specification: RunSpecification):
    pass


def construct_run_spec(
        name: str,
        parameters: dict,
        function: callable,
        codes: List[str]) -> RunDescriptor:
    """Construct a run specification."""

    parameters = Parameters.from_dict(**parameters)
    codes = CodeVersions(codes)
    return RunSpecification(
        name=name,
        parameters=parameters,
        function=function,
        codes=codes,
    )


class ComplexRunner(SimpleRunner):

    def __init__(self, cache=None):

        self.cache = cache

    def run(self, run_info: RunInfo):

        cached_run_info = self.cache.get(run_info)

        if cached_run_info is not None:
            return cached_run_info

        if run_info is None:
            parprint(f'Running {self.name}({parameters})')
            run_info = self.runner(
                self.get_wrapped_function(),
                parameters,
            )
            cache.add(run_info)
        else:
            parprint('Returning cached result for '
                     f'{self.name}({parameters})')

        code_versions = self.get_code_versions(parameters=parameters)
        # Execute the wrapped function
        # Register dependencies implement stack like data structure.
        # We register an exit handler to handle unexpected exits.
        atexit.register(clean_files, files=temporary_files)
        with dependency_stack as my_dependencies:
            with CleanEnvironment(temporary_directory) as env, \
                 clean_files(temporary_files), \
                 file_barrier(created_files, delete=False):
                tstart = time.time()
                result = self._wrapped_function(**parameters)
                tend = time.time()


        from ase.parallel import world
        metadata = dict(asr_name=self.name,
                        resources=dict(time=tend - tstart,
                                       ncores=world.size),
                        params=parameters,
                        code_versions=get_execution_info(
                            self.package_dependencies))

        # This is a hack and should be removed in the future
        # when all recipe results have been typed.
        if not isinstance(result, self.returns):
            assert isinstance(result, dict)
            result = self.returns(data=result)



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


class Runner:
    """Wrapper class for constructing recipes.

    This class implements the behaviour of an ASR recipe.

    This class wrappes a callable `func` and automatically endows the function
    with a command-line interface (CLI) through `cli` method. The CLI is
    defined using the :func:`asr.core.__init__.argument` and
    :func:`asr.core.__init__.option` functions in the core sub-package.

    The ASRCommand... XXX
    """

    package_dependencies = ('asr', 'ase', 'gpaw')

    def __init__(self, wrapped_function,
                 returns=None,
                 version=None,
                 cache=None):
        """Construct an instance of an ASRCommand.

        Parameters
        ----------
        func : callable
            Wrapped function that

        """
        assert callable(wrapped_function), \
            'The wrapped object should be callable'

        self.cache = cache

        import inspect
        mod = inspect.getmodule(wrapped_function)
        module = mod.__name__

        # Function to be executed
        self._wrapped_function = wrapped_function
        self.name = f'{module}@{wrapped_function.__name__}'

        # Return type
        if returns is None:
            returns = ASRResult
        self.returns = returns
        self.cache = cache

        # Figure out the parameters for this function
        if not hasattr(self._wrapped_function, '__asr_params__'):
            self._wrapped_function.__asr_params__ = {}

        import copy
        self.myparams = copy.deepcopy(self._wrapped_function.__asr_params__)

        import inspect
        sig = inspect.signature(self._wrapped_function)
        self.__signature__ = sig

        # Setup the CLI
        functools.update_wrapper(self, self._wrapped_function)

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

    def get_parameters(self):
        """Get the parameters of this function."""
        return self.myparams

    def cli(self, args=None):
        """Parse parameters from command line and call wrapped function.

        Parameters
        ----------
        args : List of strings or None
            List of command line arguments. If None: Read arguments from
            sys.argv.
        """
        command = setup_cli(
            self.get_wrapped_function(),
            self.main,
            self.get_defaults(),
            self.get_parameters()
        )
        return command(standalone_mode=False,
                       prog_name=f'asr run {self.name}', args=args)

    def get_wrapped_function(self):
        """Return wrapped function."""
        return self._wrapped_function

    def __call__(self, *args, **kwargs):
        """Delegate to self.main."""
        return self.main(*args, **kwargs)

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

        # Default Algorithm: construct_run_info | cache.get | dependencies.push | run | dependencies.pop | set_metadata | cache.add

        parameters = Parameters(*args, **kwargs)
        parameters = parameters.apply_defaults(self.get_signature())

        run_specification = construct_run_spec(
            function=self.get_wrapped_function(),
            parameters=parameters,
            name=self.name,
            version=self.version,
            codes=self.package_dependencies,
        )

        if self.cache.has(run_specification):
            run_record = self.cache.get(run_specification)
        else:
            with register_sideffects(run_specification) as side_effects, \
                 register_dependencies(run_specification) as dependencies, \
                 register_metadata(run_specification) as metadata:
                result = run_specification()

            run_record = construct_run_record(
                run_specification=run_specification,
                result=result,
                metadata=metadata,
                dependencies=dependencies,
                side_effects=side_effects,
            )
            self.cache.add(run_record)

        register_dependencies.register_dep(run_record)
        return run_record


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


def setup_cli(wrapped, wrapper, defparams, parameters):
    # Click CLI Interface
    CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

    cc = click.command
    co = click.option

    help = clickify_docstring(wrapped.__doc__) or ''

    command = cc(context_settings=CONTEXT_SETTINGS,
                 help=help)(wrapper)

    # Convert parameters into CLI Parameters!
    for name, param in parameters.items():
        param = param.copy()
        alias = param.pop('alias')
        argtype = param.pop('argtype')
        name2 = param.pop('name')
        assert name == name2
        assert name in parameters
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


def apply_defaults(signature, *args, **kwargs):
    """Apply defaults to args and kwargs.

    Reads the signature of the wrapped function and applies the
    defaults where relevant.

    """
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    params = copy.deepcopy(dict(bound_arguments.arguments))
    return params
