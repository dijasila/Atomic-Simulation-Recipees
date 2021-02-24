import numpy as np

from .utils import compare_equal


class NoSuchAttribute(Exception):

    pass


class Comparator:

    def __init__(self, name, function, value, *args, **kwargs):
        self.name = name
        self.function = function
        self.value = value
        self.args = args
        self.kwargs = kwargs

    def __call__(self, othervalue):
        return self.function(self.value, othervalue, *self.args, **self.kwargs)

    def __str__(self):
        return f'{self.name}({self.value})'

    def __repr__(self):
        return str(self)


def compare_any(selectors, value2):
    for comparator in selectors:
        if comparator(value2):
            return True
    else:
        return False


def match_any(*selectors):

    return Comparator(
        name='any',
        function=compare_any,
        value=selectors,
    )


def approx(value1, rtol=1e-3) -> callable:

    return Comparator(
        name='approx',
        function=np.isclose,
        value=value1,
        rtol=rtol,
    )


def equal(value1) -> callable:
    """Make comparison function that compares value with equality."""
    return Comparator(
        name='equal',
        function=compare_equal,
        value=value1,
    )


def compare_less_than(value1, value2) -> bool:
    return value1 < value2


def less_than(value1) -> callable:

    return Comparator(
        name='lessthan',
        function=compare_less_than,
        value=value1,
    )


def compare_less_than_equals(value1, value2) -> bool:
    return value1 <= value2


def less_than_equals(value1) -> callable:

    return Comparator(
        name='lessthanequals',
        function=compare_less_than_equals,
        value=value1,
    )


def compare_greater_than(value1, value2) -> bool:
    return value1 > value2


def greater_than(value1) -> callable:

    return Comparator(
        name='greaterthan',
        function=compare_greater_than,
        value=value1,
    )


def compare_greater_than_equals(value1, value2) -> bool:
    return value1 >= value2


def greater_than_equals(value1) -> callable:
    return Comparator(
        name='greaterthanequals',
        function=compare_greater_than_equals,
        value=value1,
    )


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

    return Comparator(
        name='atomsequalto',
        function=compare_atoms,
        value=atoms1,
    )


def compare_is(obj1, obj2):
    return obj1 is obj2


def check_is(obj1):

    return Comparator(
        name='atomsequalto',
        function=compare_is,
        value=obj1,
    )


def compare_contains(obj1, obj2):
    return obj1 in obj2


def contains(obj1):

    return Comparator(
        name='contains',
        function=compare_contains,
        value=obj1,
    )


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
    ANY = staticmethod(match_any)
    CONTAINS = staticmethod(contains)

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
        parts = [
            f'{key}={repr(value)}'
            for key, value in
            self.selection.items()
        ]
        return 'Selector(' + ', '.join(parts) + ')'

    def __repr__(self):
        return self.__str__()


def get_attribute(obj, attrs):

    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        elif attr in obj:
            obj = obj[attr]
        else:
            raise NoSuchAttribute

    return obj
