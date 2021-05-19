"""Implement parameter handling."""

import pathlib
import typing
import contextlib
import copy

from asr.core import read_json, get_recipe_from_name
from .utils import compare_equal


def fill_in_defaults(dct, defaultdct):
    """Fill dct None entries with values from defaultdct."""
    new_dct = {}

    for key, value in dct.items():
        if key in [..., None]:
            for key in defaultdct:
                if key not in dct:
                    new_dct[key] = defaultdct[key]
        else:
            if isinstance(value, dict):
                new_dct[key] = fill_in_defaults(value, defaultdct.get(key, {}))
            else:
                new_dct[key] = value
    return new_dct


PARAMETERS = {}


@contextlib.contextmanager
def set_defaults(parameters: typing.Dict[str, typing.Any]):  # noqa
    defaults = {}
    for name in parameters:
        recipe = get_recipe_from_name(name)
        defaults[name] = recipe.defaults

    parameters = fill_in_defaults(parameters, defaults)
    prev_params = copy.deepcopy(PARAMETERS)
    PARAMETERS.update(parameters)
    yield

    keys = list(PARAMETERS.keys())
    for key in keys:
        del PARAMETERS[key]
    PARAMETERS.update(prev_params)


def get_default_parameters(name, list_of_defaults=None):  # noqa

    if list_of_defaults is None:
        list_of_defaults = [PARAMETERS]
        paramsfile = pathlib.Path('params.json')
        if paramsfile.is_file():
            list_of_defaults.append(read_json(paramsfile))

    for defaults in list_of_defaults:
        if name in defaults:
            return defaults[name]

    return {}


class Parameters:  # noqa

    def __init__(self, parameters: typing.Dict[str, typing.Any]):  # noqa
        self.__dict__.update(parameters)

    def __hash__(self):
        """Make parameter hash."""
        return hash(self.__dict__)

    def keys(self):  # noqa
        return self.__dict__.keys()

    def __getitem__(self, key):
        """Get parameter."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def update(self, parameters: 'Parameters'):
        self.__dict__.update(parameters.__dict__)

    def items(self):  # noqa
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __str__(self):  # noqa
        return ','.join([f'{key}={value}' for key, value in self.__dict__.items()])

    def __repr__(self):  # noqa
        return 'Parameters(' + str(self.__dict__) + ')'

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def copy(self):
        return Parameters(copy.deepcopy(self.__dict__))

    def __eq__(self, other):
        if not isinstance(other, Parameters):
            return False
        return compare_equal(self.__dict__, other.__dict__)

    def __contains__(self, key):
        return key in self.__dict__

    def __delitem__(self, item):
        del self.__dict__[item]
