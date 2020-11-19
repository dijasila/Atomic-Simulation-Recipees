import pathlib
import typing
import contextlib
import copy
from asr.core import read_json, get_recipe_from_name


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
def set_defaults(parameters: typing.Dict[str, typing.Any]):
    defaults = {}
    for name in parameters:
        recipe = get_recipe_from_name(name)
        defaults[name] = recipe.get_defaults()

    parameters = fill_in_defaults(parameters, defaults)
    prev_params = copy.deepcopy(PARAMETERS)
    PARAMETERS.update(parameters)
    yield

    keys = list(PARAMETERS.keys())
    for key in keys:
        del PARAMETERS[key]
    PARAMETERS.update(prev_params)


def get_default_parameters(name, list_of_defaults=None):

    if list_of_defaults is None:
        list_of_defaults = [PARAMETERS]
        paramsfile = pathlib.Path('params.json')
        if paramsfile.is_file():
            list_of_defaults.append(read_json(paramsfile))

    for defaults in list_of_defaults:
        if name in defaults:
            return defaults[name]

    return {}
