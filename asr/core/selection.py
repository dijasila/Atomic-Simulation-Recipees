import functools
import numpy as np
from ase import Atoms

from .params import Parameters


class NoSuchAttribute(Exception):

    pass


def flatten_list(lst):
    flatlist = []
    for value in lst:
        if isinstance(value, list):
            flatlist.extend(flatten_list(value))
        else:
            flatlist.append(value)
    return flatlist


def compare_lists(lst1, lst2):
    flatlst1 = flatten_list(lst1)
    flatlst2 = flatten_list(lst2)
    if not len(flatlst1) == len(flatlst2):
        return False
    for value1, value2 in zip(flatlst1, flatlst2):
        if not value1 == value2:
            return False
    return True


def compare_ndarrays(array1, array2):
    lst1 = array1.tolist()
    lst2 = array2.tolist()
    return compare_lists(lst1, lst2)


def approx(value1, rtol=1e-3):

    def wrapped_approx(value2):
        return np.isclose(value1, value2, rtol=rtol)

    return wrapped_approx


def compare_atoms(atoms1, atoms2):
    dct1 = atoms1.todict()
    dct2 = atoms2.todict()
    keys = [
        'numbers',
        'positions',
        'cell',
        'pbc',
    ]
    for key in keys:
        if not np.allclose(dct1[key], dct2[key]):
            return False
    return True


def allways_match(value):
    return True


class Selection:

    def __init__(self, **selection):
        self.selection = self.normalize_selection(selection)

    def do_not_compare(self, x):
        return True

    def normalize_selection(self, selection: dict):
        normalized = {}

        for key, value in selection.items():
            comparator = None
            if isinstance(value, dict):
                norm = self.normalize_selection(value)
                for keynorm, valuenorm in norm.items():
                    normalized['.'.join([key, keynorm])] = valuenorm
            elif value is None:
                pass
            elif isinstance(value, Atoms):
                comparator = functools.partial(compare_atoms, value)
            elif isinstance(value, np.ndarray):
                comparator = functools.partial(
                    compare_ndarrays,
                    value,
                )
            elif isinstance(value, (list, tuple)):
                comparator = functools.partial(
                    compare_lists,
                    value,
                )
            elif type(value) in {str, bool, int}:
                comparator = value.__eq__
            elif type(value) is float:
                comparator = approx(value)
            elif ((hasattr(value, '__dict__') and value.__dict__)
                  or isinstance(value, Parameters)):
                norm = self.normalize_selection(value.__dict__)
                for keynorm, valuenorm in norm.items():
                    normalized['.'.join([key, keynorm])] = valuenorm
            elif callable(value):
                comparator = value
            # else: XXX Make special comparator type.
            #     raise AssertionError(f'Unknown type {type(value)}')

            if comparator is not None:
                normalized[key] = comparator
        return normalized

    def get_attribute(self, obj, attrs):

        if not attrs:
            return obj

        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            elif attr in obj:
                obj = obj[attr]
            else:
                raise NoSuchAttribute

        return obj

    def matches(self, obj):

        for attr, comparator in self.selection.items():
            try:
                objvalue = self.get_attribute(obj, attr.split('.'))
            except NoSuchAttribute:
                return False
            if not comparator(objvalue):
                return False
        return True

    def __str__(self):
        return str(self.selection)
