import typing
import inspect
from .comparators import comparators
from .command import ASRCommand
from asr.calculators import set_calculator_hook


def instruction(
        module: typing.Optional[str] = None,
        version: int = 0,
        argument_hooks: typing.List[typing.Callable] = [set_calculator_hook],
        package_dependencies=('asr', 'ase', 'gpaw'),
):
    """Make instruction object.

    Parameters
    ----------
    module: str
        Name of recipe that instruction belongs to.
    version: int
        Instruction version number.
    argument_hooks: list[callable]
        Functions for preprocessing arguments. Gets a parameter object and should
        return parameter object.
    package_dependencies: typing.Tuple[str]
        Python packages for which versions should be logged.
    """

    def decorator(func):
        if module is None:
            mod = inspect.getmodule(func)
            mod = mod.__name__
        else:
            mod = module

        return ASRCommand(
            func,
            module=mod,
            version=version,
            argument_hooks=argument_hooks,
            package_dependencies=package_dependencies,
        )

    return decorator


command = instruction


def option(*aliases, matcher=comparators.EQUAL, **kwargs):
    """Make argument descriptor for a CLI option.

    Parameters
    ----------
    aliases: list[str]
    matcher: callable
        Matching function that wraps value when matching parameter with cache.
        Default is to use "equal", eg. EQUAL("value").
    kwargs: dict
        Arguments forwarded to click.option.
    """

    def decorator(func):
        assert aliases, 'You have to give a name to this parameter'

        for arg in aliases:
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
                 'alias': aliases,
                 'name': name,
                 'matcher': matcher}
        param.update(kwargs)
        _add_param(func, param)
        return func

    return decorator


def argument(name, matcher=comparators.EQUAL, **kwargs):
    """Make argument descriptor for a CLI argument.

    Parameters
    ----------
    name: str
        Name of argument.
    matcher: callable
        Matching function that wraps value when matching parameter with cache.
        Default is to use "equal", eg. EQ("value").
    kwargs: dict
        Arguments forwarded to click.argument.
    """

    def decorator(func):
        assert 'default' not in kwargs, 'Arguments do not support defaults!'
        param = {'argtype': 'argument',
                 'alias': (name, ),
                 'name': name,
                 'matcher': matcher}
        param.update(kwargs)
        _add_param(func, param)
        return func

    return decorator


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
