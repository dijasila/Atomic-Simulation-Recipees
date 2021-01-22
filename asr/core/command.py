"""Implement ASRCommand class and related decorators."""
from . import (
    clickify_docstring,
    ASRResult,
)
import functools
import click
import copy
import inspect
import typing
from .cache import get_cache
from .parameters import get_default_parameters, Parameters
from .record import RunRecord
from .specification import construct_run_spec
from .workdir import isolated_work_dir
from .results import obj_to_id
from .dependencies import register_dependencies
from .resources import register_resources
from .cache import Cache
from .selector import Selector


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

    def __init__(
            self,
            wrapped_function,
            module=None,
            returns=None,
            version=0,
            cache=None,
            argument_hooks=None,
            migrations=None,
    ):
        """Construct an instance of an ASRCommand.

        Parameters
        ----------
        func : callable
            Wrapped function that

        """
        assert callable(wrapped_function), \
            'The wrapped object should be callable'

        if cache is None:
            cache = get_cache(backend='filesystem')
        self.cache = cache
        self._migrations = migrations
        self.version = version
        if argument_hooks is None:
            self.argument_hooks = []
        else:
            self.argument_hooks = argument_hooks
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

    def migrations(self, cache):
        if self._migrations:
            return self._migrations(cache)

    def get_signature(self):
        """Return signature with updated defaults based on params.json."""
        myparams = []
        for key, value in self.__signature__.parameters.items():
            assert key in self.myparams, \
                f'Missing description for param={key},value={value}.'
            myparams.append(key)

        # Check that all annotated parameters can be found in the
        # actual function signature.
        myparams = [k for k, v in self.__signature__.parameters.items()]
        for key in self.myparams:
            assert key in myparams, f'param={key} is unknown.'

        default_parameters = get_default_parameters(self.name)
        if default_parameters:
            signature_parameters = dict(self.__signature__.parameters)
            for key, new_default in default_parameters.items():
                assert key in signature_parameters, \
                    f'Unknown param in params.json: param={key}.'
                parameter = signature_parameters[key]
                signature_parameters[key] = parameter.replace(
                    default=new_default)

            new_signature = self.__signature__.replace(
                parameters=[val for val in signature_parameters.values()])
            return new_signature

        return self.__signature__

    @property
    def defaults(self):
        """Get default parameters based on signature and params.json."""
        signature = self.get_signature()
        defparams = {}
        for key, value in signature.parameters.items():
            if value.default is not inspect.Parameter.empty:
                defparams[key] = value.default
        return Parameters(parameters=defparams)

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
        command = make_cli_command(self)
        return command(standalone_mode=False,
                       prog_name=f'asr run {self.name}', args=args)

    def get_wrapped_function(self):
        """Return wrapped function."""
        return self._wrapped_function

    def __call__(self, *args, **kwargs):
        """Delegate to self.main."""
        return self.main(*args, **kwargs)

    def make_selector(
            self,
            cache: Cache = None,
            selector: Selector = None,
            equals={},
    ) -> Selector:

        if cache is None:
            cache = self.cache

        selector = cache.make_selector(
            selector=selector,
            equals=equals,
        )

        selector.run_specification.name = selector.EQ(
            obj_to_id(self.get_wrapped_function())
        )
        return selector

    def get(self,
            cache: typing.Optional[Cache] = None,
            selector: typing.Optional[Selector] = None,
            **equals):

        if cache is None:
            cache = self.cache

        selector = self.make_selector(
            cache=cache,
            selector=selector,
            equals=equals,
        )

        return cache.get(selector=selector)

    def select(
            self,
            selector: typing.Optional[Selector] = None,
            cache: typing.Optional[Cache] = None,
            **equals,

    ):
        if cache is None:
            cache = self.cache

        selector = self.make_selector(
            cache=cache,
            selector=selector,
            equals=equals,
        )

        return cache.select(selector=selector)

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

        # TODO: The caching database could be of a non-relational
        # format (would be similar to current format).

        # TODO: Should Result objects have some sort of verification
        # mechanism? Like checking acoustic sum rules?

        # TODO: Make clean run environment class?
        # TODO: Make an old type results object.

        # REQ: Recipe must be able to run multiple times and cache
        # their results (think LRU-cache).

        # REQ: Must be possible to change implementation of recipe
        #      without invalidating previous results
        # REQ: Must support side-effects such as files written to disk.
        # REQ: Must store information about code versions
        # REQ: Must be able to read defaults from configuration file on
        #      a per-folder basis.
        # REQ: Must support chaining of recipes (dependencies).

        # REQ: Caching database should be simple and decentralized
        # (think sqlite).

        # REQ: Caching database should be plain text.

        # REQ: Returned object should be self-contained (think ase
        # BandStructure object).

        # REQ: Returned objects must be able to present themselves as
        # figures and HTML.

        # REQ: Must be delocalized from ASR (ie. must be able to have
        # a seperate set of recipes locally, non-related to asr).

        # REQ: Must also have a packaging mechanism for entire
        # projects (ie. ASE databases).

        # REQ: Must be possible to call as a simple python function
        # thus circumventing CLI.

        # REQ: Must be able to call without scripting, eg. through a CLI.
        # REQ: Must support all ASE calculators.

        parameters = apply_defaults(
            self.get_signature(), *args, **kwargs)
        parameters = Parameters(parameters=parameters)
        for hook in self.argument_hooks:
            parameters = hook(parameters)

        run_specification = construct_run_spec(
            name=obj_to_id(self.get_wrapped_function()),
            parameters=parameters,
            version=self.version,
            codes=self.package_dependencies,
        )

        cache = self.cache

        @register_dependencies.register
        @cache()
        @register_dependencies()
        @isolated_work_dir()
        @register_resources()
        def execute_run_spec(run_spec):
            name = run_spec.name
            parameters = run_spec.parameters
            paramstring = ', '.join([f'{key}={repr(value)}' for key, value in
                                     parameters.items()])
            print(f'Running {name}({paramstring})')
            result = run_spec()
            run_record = RunRecord(
                result=result,
                run_specification=run_spec,
            )
            return run_record

        run_record = execute_run_spec(run_specification)
        return run_record


def command(*decoargs, **decokwargs):

    def decorator(func):
        return ASRCommand(func, *decoargs, **decokwargs)

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


def make_cli_command(asr_command: ASRCommand):
    command = setup_cli(
        asr_command.get_wrapped_function(),
        asr_command.main,
        asr_command.defaults,
        asr_command.get_parameters(),
    )
    return command


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

        if 'type' in param:
            try:
                param['type'].default = default
            except (AttributeError, TypeError):
                pass

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
    bound_arguments = signature.bind_partial(*args, **kwargs)
    bound_arguments.apply_defaults()
    params = copy.deepcopy(dict(bound_arguments.arguments))
    return params
