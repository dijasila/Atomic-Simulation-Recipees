"""Implement ASRCommand class and related decorators."""
from . import (
    read_json,
    md5sum,
    clickify_docstring,
    ASRResult,
    chdir,
    write_file,
    # file_barrier,
)
# import os
import contextlib
import functools
import abc
# from .temporary_directory import temporary_directory
# from .dependencies import dependency_stack
# from .cache import ASRCache
import typing
# from ase.parallel import parprint
# import atexit
import click
import copy
# import time
from importlib import import_module
from pathlib import Path
import inspect
import json
from asr.core.results import get_object_matching_obj_id


class Parameter:

    def __init__(self, name, value, hash_func):

        self.name = name
        self.value = value
        self.hash_func = hash_func


class Parameters:

    def __init__(self, parameters: typing.Dict[str, Parameter]):
        self._parameters = parameters

    def __hash__(self):
        """Make parameter hash."""
        return hash(self._parameters)

    def keys(self):
        return self._parameters.keys()

    def __getitem__(self, key):
        """Get parameter."""
        return self._parameters[key]

    def __getattr__(self, key):
        """Get parameter."""
        if key not in self._parameters:
            raise AttributeError
        return self._parameters[key]

    def __str__(self):
        return str(self._parameters)

    def __repr__(self):
        return self.__str__()


class RunSpecification:

    spec_version: int = 0

    def __init__(
            self,
            name: str,
            parameters: Parameters,
            version: int,
            # codes: CodeDescriptor,
    ):
        self.parameters = parameters
        self.name = name
        # self.codes = codes
        self.version = version

    def __call__(
            self,
    ):
        obj = get_object_matching_obj_id(self.name)
        function = obj.get_wrapped_function()
        return function(**self.parameters)

    def __str__(self):
        text = [
            '',
            'RunSpecification',
            f'    name={self.name}',
            f'    parameters={self.parameters}',
            f'    version={self.version}',
        ]
        return '\n'.join(text)

    def __repr__(self):
        return self.__str__()


class RunRecord:

    record_version: int = 0

    def __init__(
            self,
            run_specification: RunSpecification,
            result: typing.Any,
            # metadata: MetaData,
            side_effects: 'SideEffects',
            record_id: typing.Optional[str] = None,
            # dependencies: Dependencies,
    ):
        self._run_specification = run_specification
        self._result = result
        self._side_effects = side_effects
        self._record_id = record_id

    @property
    def result(self):
        return self._result

    @property
    def parameters(self):
        return self._run_specification.parameters

    @property
    def side_effects(self):
        return self._side_effects

    @property
    def run_specification(self):
        return self._run_specification

    @property
    def record_id(self):
        return self._record_id

    @record_id.setter
    def record_id(self, value):
        assert self.record_id is None, 'Record id was already set.'
        self._record_id = value

    def __str__(self):
        text = [
            '',
            'RunRecord',
            f'    run_specification={self.run_specification}',
            f'    parameters={self.parameters}',
            f'    side_effects={self.side_effects}',
            f'    result={self.result:str}',
            f'    record_id={self.record_id}',
        ]
        return '\n'.join(text)

    def __repr__(self):
        return self.__str__()


@contextlib.contextmanager
def register_metadata(run_specification: RunSpecification):
    metadata = {}
    yield metadata


class SideEffect:

    def __init__(self, filename):

        self.filename
        self.hash = None


side_effects_stack = []


class RegisterSideEffects():

    def __init__(self, side_effects_stack=side_effects_stack):
        self.side_effects_stack = side_effects_stack
        self._root_dir = None

    def get_workdir_name(self, root_dir, run_specification: RunSpecification) -> Path:
        run_number = 1
        workdirformat = '.asr/{run_specification.name}{}'
        while (root_dir / workdirformat.format(
                run_number,
                run_specification=run_specification)).is_dir():
            run_number += 1

        workdir = root_dir / workdirformat.format(
            run_number,
            run_specification=run_specification)

        return workdir

    def __enter__(self):
        """Append empty side effect object to stack."""
        side_effects = {}
        self.side_effects_stack.append(side_effects)
        return side_effects

    def __exit__(self, type, value, traceback):
        """Register side effects and pop side effects from stack."""
        side_effects = self.side_effects_stack[-1]
        side_effects.update(
            {
                path.name: str(path.absolute())
                for path in Path().glob('*')
            }
        )
        self.side_effects_stack.pop()

    def make_decorator(self, run_specification):

        def decorator(func):
            def wrapped(*args, **kwargs):
                current_dir = Path().absolute()
                if self._root_dir is None:
                    self._root_dir = current_dir

                workdir = self.get_workdir_name(self._root_dir, run_specification)
                side_effects = {}
                with chdir(workdir, create=True):
                    with self as side_effects:
                        result = func(*args, **kwargs)
                        result = {'side_effects': side_effects, **result}

                if not self.side_effects_stack:
                    self._root_dir = None
                return result
            return wrapped

        return decorator

    def __call__(self, run_specification):
        return self.make_decorator(run_specification)


def construct_run_record(
        run_specification: RunSpecification,
        result: typing.Any,
        side_effects: typing.Dict[str, str],
):
    return RunRecord(run_specification,
                     result=result,
                     side_effects=side_effects)


# class Code:
#     pass


# class Codes:

#     def __init__(*codes: typing.List[Code]):
#         pass


def construct_run_spec(
        name: str,
        parameters: typing.Union[dict, Parameters],
        version: int,
        # codes: typing.Union[typing.List[str], Codes],
) -> RunSpecification:
    """Construct a run specification."""
    if not isinstance(parameters, Parameters):
        parameters = Parameters.from_dict(**parameters)

    # if not isinstance(codes, Codes):
    #     codes = Codes.from_list(codes)

    return RunSpecification(
        name=name,
        parameters=parameters,
        version=version,
        # codes=codes,
    )


# class ComplexRunner(SimpleRunner):

#     def __init__(self, cache=None):

#         self.cache = cache

#     def run(self, run_info: RunInfo):

#         cached_run_info = self.cache.get(run_info)

#         if cached_run_info is not None:
#             return cached_run_info

#         if run_info is None:
#             parprint(f'Running {self.name}({parameters})')
#             run_info = self.runner(
#                 self.get_wrapped_function(),
#                 parameters,
#             )
#             cache.add(run_info)
#         else:
#             parprint('Returning cached result for '
#                      f'{self.name}({parameters})')

#         code_versions = self.get_code_versions(parameters=parameters)
#         # Execute the wrapped function
#         # Register dependencies implement stack like data structure.
#         # We register an exit handler to handle unexpected exits.
#         atexit.register(clean_files, files=temporary_files)
#         with dependency_stack as my_dependencies:
#             with CleanEnvironment(temporary_directory) as env, \
#                  clean_files(temporary_files), \
#                  file_barrier(created_files, delete=False):
#                 tstart = time.time()
#                 result = self._wrapped_function(**parameters)
#                 tend = time.time()


#         from ase.parallel import world
#         metadata = dict(asr_name=self.name,
#                         resources=dict(time=tend - tstart,
#                                        ncores=world.size),
#                         params=parameters,
#                         code_versions=get_execution_info(
#                             self.package_dependencies))

#         # This is a hack and should be removed in the future
#         # when all recipe results have been typed.
#         if not isinstance(result, self.returns):
#             assert isinstance(result, dict)
#             result = self.returns(data=result)

class AbstractCache(abc.ABC):

    @abc.abstractmethod
    def add(self, run_record: RunRecord):
        pass

    @abc.abstractmethod
    def get(self, run_record: RunRecord):
        pass

    @abc.abstractmethod
    def has(self, run_specification: RunSpecification):
        pass


class NoCache(AbstractCache):

    def add(self, run_record: RunRecord):
        """Add record of run to cache."""
        ...

    def has(self, run_specification: RunSpecification) -> bool:
        """Has run record matching run specification."""
        return False

    def get(self, run_specification: RunSpecification):
        """Get run record matching run specification."""
        ...


class RunSpecificationAlreadyExists(Exception):
    pass


class Serializer(abc.ABC):

    @abc.abstractmethod
    def serialize(obj: typing.Any) -> str:
        pass

    @abc.abstractmethod
    def deserialize(serialized: str) -> typing.Any:
        pass


from asr.core.results import obj_to_id
from numpy import ndarray
import ase.io.jsonio


class ASRJSONEncoder(json.JSONEncoder):

    def default(self, obj) -> dict:

        try:
            return ase.io.jsonio.MyEncoder.default(self, obj)
        except TypeError:
            pass
        if hasattr(obj, '__dict__'):
            cls_id = obj_to_id(obj.__class__)
            # dct = {}
            # for key, value in obj.__dict__.items():
            #     dct[key] = self.default(value)
            obj = {'cls_id': cls_id, '__dict__':
                   copy.copy(obj.__dict__)}

            return obj
        return json.JSONEncoder.default(self, obj)
        # obj_type = type(obj)

        # if obj_type == dict:
        #     for key, value in obj.items():
        #         obj[key] = self.default(value)
        #     return obj
        # elif obj_type in [tuple, list]:
        #     obj = list(obj)
        #     for i, value in enumerate(obj):
        #         obj[i] = self.default(value)
        #     return obj
        # elif obj_type in {str, int, float, bool, type(None)}:
        #     return obj




def json_hook(json_object: dict):
    from asr.core.results import get_object_matching_obj_id

    if 'cls_id' in json_object:
        cls = get_object_matching_obj_id(json_object['cls_id'])
        obj = cls.__new__(cls)
        obj.__dict__.update(json_object['__dict__'])
        return obj

    return json_object


class JSONSerializer(Serializer):

    encoder = ASRJSONEncoder().encode
    decoder = json.JSONDecoder(object_hook=json_hook).decode
    accepted_types = {dict, list, str, int, float, bool, type(None)}

    def serialize(self, obj) -> str:
        """Serialize object to JSON."""
        return self.encoder(obj)

    def deserialize(self, serialized: str) -> typing.Any:
        """Deserialize json object."""
        return self.decoder(serialized)


class SingleRunFileCache(AbstractCache):

    def __init__(self, serializer: Serializer = JSONSerializer()):
        self.serializer = serializer
        self._cache_dir = None
        self.depth = 0

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value):
        self._cache_dir = value

    @staticmethod
    def _name_to_results_filename(name: str):
        name = name.replace('::', '@').replace('@main', '')
        return f'results-{name}.json'

    def add(self, run_record: RunRecord):
        if self.has(run_record.run_specification):
            raise RunSpecificationAlreadyExists(
                'You are using the SingleRunFileCache which does not'
                'support multiple runs of the same function. '
                'Please specify another cache.'
            )
        name = run_record.run_specification.name
        filename = self._name_to_results_filename(name)
        serialized_object = self.serializer.serialize(run_record)
        self._write_file(filename, serialized_object)

    def has(self, run_specification: RunSpecification):
        name = run_specification.name
        filename = self._name_to_results_filename(name)
        return (self.cache_dir / filename).is_file()

    def get(self, run_specification: RunSpecification):
        name = run_specification.name
        filename = self._name_to_results_filename(name)
        serialized_object = self._read_file(filename)
        obj = self.serializer.deserialize(serialized_object)
        return obj

    def _write_file(self, filename: str, text: str):
        write_file(self.cache_dir / filename, text)

    def _read_file(self, filename: str) -> str:
        serialized_object = Path(self.cache_dir / filename).read_text()
        return serialized_object

    def __enter__(self):
        """Enter context manager."""
        if self.depth == 0:
            self.cache_dir = Path('.').absolute()
        self.depth += 1
        return self

    def __exit__(self, type, value, traceback):
        """Exit context manager."""
        self.depth -= 1
        if self.depth == 0:
            self.cache_dir = None

    def __call__(self, run_specification: RunSpecification):

        def wrapper(func):

            def wrapped(*args, **kwargs):
                with self:
                    if self.has(run_specification):
                        run_record = self.get(run_specification)
                    else:
                        run_data = func(*args, **kwargs)
                        run_record = construct_run_record(**run_data)
                        self.add(run_record)
                return run_record
            return wrapped
        return wrapper


def to_json(obj):
    """Write an object to a json file."""
    json_string = obj.format_as('json')
    return json_string


def get_md5_checksums(filenames: typing.List[str]) -> typing.Dict[str, str]:
    """Get md5 checksums of a list of files."""
    checksums = {}
    for filename in filenames:
        hexdigest = md5sum(filename)
        checksums[filename] = hexdigest
    return checksums


def does_files_exist(filenames: typing.List[str]) -> typing.List[bool]:
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


dependency_stack = []


class RegisterDependencies:

    def __init__(self, dependency_stack=dependency_stack):
        self.dependency_stack = dependency_stack

    def __enter__(self):
        """Add frame to dependency stack."""
        dependencies = []
        self.dependency_stack.append(dependencies)
        return dependencies

    def __exit__(self, type, value, traceback):
        """Pop frame of dependency stack."""
        self.dependency_stack.pop()

    def __call__(self, run_specification: RunSpecification):

        def wrapper(func):

            def wrapped(*args, **kwargs):

                with self as dependencies:
                    result = func(*args, **kwargs)
                result = {'dependencies': dependencies, **result}

                return result
            return wrapped
        return wrapper


register_dependencies = RegisterDependencies()


# def register_dependencies(run_specification: RunSpecification):
#     """Register dependencies."""

#     def wrapper(func):

#         def wrapped(*args, **kwargs):

#             with RegisterDependencies(run_specification) as dependencies:
#                 result = func(*args, **kwargs)
#             result = {'dependencies': dependencies, **result}

#             return result
#         return wrapped
#     return wrapper


register_side_effects = RegisterSideEffects()

# def register_side_effects(run_specification: RunSpecification):

#     def wrapper(func):

#         def wrapped(*args, **kwargs):
#             with RegisterSideEffects(run_specification) as side_effects:
#                 result = func(*args, **kwargs)
#             result = {'side_effects': side_effects, **result}
#             return result

#         return wrapped

#     return wrapper


def register_run_spec(run_specification):

    def wrapper(func):

        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            result = {'run_specification': run_specification, **result}
            return result

        return wrapped

    return wrapper


single_run_file_cache = SingleRunFileCache()

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
            cache=single_run_file_cache,
            dependencies=None,
            creates=None,
            requires=None,
            resources=None,
            tests=None,
            save_results_file=None,
    ):
        """Construct an instance of an ASRCommand.

        Parameters
        ----------
        func : callable
            Wrapped function that

        """
        assert callable(wrapped_function), \
            'The wrapped object should be callable'

        self.cache = cache
        self.version = version

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

        parameters = apply_defaults(self.get_signature(), *args, **kwargs)
        parameters = Parameters(parameters=parameters)

        run_specification = construct_run_spec(
            name=obj_to_id(self.get_wrapped_function()),
            parameters=parameters,
            version=self.version,
            # codes=self.package_dependencies,
        )

        # with self.cache as cache:
        #     if cache.has(run_specification):
        #         run_record = self.cache.get(run_specification)
        #     else:
        cache = self.cache

        # @emit_dependency(run_specification)
        @cache(run_specification)
        # @register_dependencies(run_specification)
        @register_side_effects(run_specification)
        @register_run_spec(run_specification)
        # @register_metadata(run_specification)
        def execute_run_spec():
            result = run_specification()
            return {'result': result}

        run_record = execute_run_spec()
        # run_record = construct_run_record(**run_data)
        # cache_id = self.cache.add(run_record)
        # register_dependencies.register_dep(run_record)
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


def command(*decoargs, **decokwargs):

    print(decoargs, decokwargs)

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
