import functools
import numpy as np
from ase import Atoms


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


def compare_lists(lst1, lst2) -> bool:
    flatlst1 = flatten_list(lst1)
    flatlst2 = flatten_list(lst2)
    if not len(flatlst1) == len(flatlst2):
        return False
    for value1, value2 in zip(flatlst1, flatlst2):
        if not value1 == value2:
            return False
    return True


def compare_ndarrays(array1, array2) -> bool:
    lst1 = array1.tolist()
    lst2 = array2.tolist()
    return compare_lists(lst1, lst2)


def approx(value1, rtol=1e-3) -> callable:

    def wrapped_approx(value2) -> bool:
        return np.isclose(value1, value2, rtol=rtol)

    return wrapped_approx


def equal(value1) -> callable:
    """Make comparison function that compares value with equality."""

    def wrapped_equal(value2) -> bool:
        return compare_equal(value1, value2)

    return wrapped_equal


def less_than(value1) -> callable:

    def wrapped_less_than(value2) -> bool:
        return value1 < value2

    return wrapped_less_than


def less_than_equals(value1) -> callable:

    def wrapped_less_than_equals(value2) -> bool:
        return value1 <= value2

    return wrapped_less_than_equals


def greater_than(value1) -> callable:

    def wrapped_greater_than(value2) -> bool:
        return value1 > value2

    return wrapped_greater_than


def greater_than_equals(value1) -> callable:

    def wrapped_greater_than_equals(value2) -> bool:
        return value1 >= value2

    return wrapped_greater_than_equals


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


def atoms_equal_to(atoms1):

    def wrapped_atoms_equal_to(atoms1, atoms2):
        return compare_atoms(atoms1, atoms2)

    return wrapped_atoms_equal_to


def check_is(obj1):

    def wrapped_is(obj1, obj2):
        return obj1 is obj2

    return wrapped_is


def allways_match(value):
    return True


class SelectorSetter:

    def __init__(self, selection: 'Selector', attr):
        self.__dict__['selection'] = selection
        self.__dict__['attrs'] = [attr]

    def __getattr__(self, attr):
        self.attrs.append(attr)
        return self

    def __setattr__(self, attr, value):
        setattr(
            self.selection,
            '.'.join(self.attrs + [attr]),
            value,
        )


class Selector:

    # Shortcuts for comparison functions
    EQ = EQUAL = staticmethod(equal)
    IS = staticmethod(check_is)
    LT = LESS_THAN = staticmethod(less_than)
    GT = GREATER_THAN = staticmethod(greater_than)
    LTE = LESS_THAN_EQUALS = staticmethod(less_than_equals)
    GTE = GREATER_THAN_EQUALS = staticmethod(greater_than_equals)
    APPROX = staticmethod(approx)
    ATOMS_EQUAL_TO = staticmethod(atoms_equal_to)

    def __init__(self, **selection):
        self.__dict__['selection'] = {}

        for key, value in selection.items():
            setattr(self, key, value)

    def matches(self, obj) -> bool:

        for attr, comparator in self.selection.items():
            try:
                objvalue = get_attribute(obj, attr.split('.'))
            except NoSuchAttribute:
                return False
            if not comparator(objvalue):
                return False
        return True

    def __setattr__(self, attr, value):
        self.selection[attr] = value

    def __getattr__(self, attr):
        return SelectorSetter(self, attr)

    def __str__(self):
        return str(self.selection)

    def __repr__(self):
        return self.__str__()


def compare_equal(obj1, obj2):

    try:
        return bool(obj1 == obj2)
    except ValueError:
        if not type(obj1) == type(obj2):
            return False
        if type(obj1) is np.ndarray:
            return compare_ndarrays(obj1, obj2)
        raise


def normalize_selection(selection: dict):
    normalized = {}

    for key, value in selection.items():
        comparator = None
        if isinstance(value, dict):
            norm = normalize_selection(value)
            for keynorm, valuenorm in norm.items():
                normalized['.'.join([key, keynorm])] = valuenorm
        elif isinstance(value, (list, tuple, str, bool, int, complex,
                                float, Atoms, np.ndarray)):
            comparator = functools.partial(compare_equal, value)
        # elif ((hasattr(value, '__dict__') and value.__dict__)
        #       or isinstance(value, Parameters)):
        #     norm = normalize_selection(value.__dict__)
        #     for keynorm, valuenorm in norm.items():
        #         normalized['.'.join([key, keynorm])] = valuenorm
        # elif isinstance(value, Comparator):  # callable(value):
        #     comparator = value
        # else: XXX Make special comparator type.
        #     raise AssertionError(f'Unknown type {type(value)}')

        if comparator is not None:
            normalized[key] = comparator
    return normalized


def get_attribute(obj, attrs):

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
