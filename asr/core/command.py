"""Implement ASRCommand class and related decorators."""
from . import (read_json, write_file, md5sum,
               file_barrier, clickify_docstring, ASRResult,
               get_recipe_from_name)
import functools
from ase.parallel import parprint
import click
import copy
import time
from importlib import import_module
from pathlib import Path
import inspect


def to_json(obj):
    """Write an object to a json file."""
    json_string = obj.format_as('json')
    return json_string


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
                 dependencies=None,
                 creates=None,
                 returns=None,
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

            recipe()

        assert self.is_requirements_met(), \
            (f'{self.name}: Some required files are missing: {self.requires}. '
             'This could be caused by incorrect dependencies.')

        # Use the wrapped functions signature to create dictionary of
        # parameters
        signature = self.get_signature()
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        params = dict(bound_arguments.arguments)

        paramstring = ', '.join([f'{key}={repr(value)}' for key, value in
                                 params.items()])
        parprint(f'Running {self.name}({paramstring})')

        # Execute the wrapped function
        with file_barrier(self.created_files, delete=False):
            tstart = time.time()
            result = self._main(**copy.deepcopy(params)) or {}
            tend = time.time()

        if not isinstance(result, self.returns):
            assert isinstance(result, dict)
            result = self.returns(data=result)

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
