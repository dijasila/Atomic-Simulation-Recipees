"""Implements comparators used for matching objects."""
import types
import numpy as np
from .utils import compare_equal


class Comparator:
    """Class that represents a comparison function."""

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


def ANY(*selectors):

    return Comparator(
        name='any',
        function=compare_any,
        value=selectors,
    )


def compare_all(comparators, value2):
    for comparator in comparators:
        if not comparator(value2):
            return False
    else:
        return True


def ALL(*comparators):
    return Comparator(
        name='all',
        function=compare_all,
        value=comparators,
    )


def OR(comparator1, comparator2):

    return Comparator(
        name='or',
        function=compare_any,
        value=[comparator1, comparator2],
    )


def AND(comparator1, comparator2):
    return Comparator(
        name='and',
        function=compare_all,
        value=[comparator1, comparator2],
    )


def compare_not(comparator, value):
    return not comparator(value)


def NOT(comparator):
    return Comparator(
        name='not',
        function=compare_not,
        value=comparator,
    )


def APPROX(value1, rtol=1e-3) -> Comparator:

    return Comparator(
        name='approx',
        function=np.isclose,
        value=value1,
        rtol=rtol,
    )


def EQUAL(value1) -> Comparator:
    """Make comparison function that compares value with equality."""
    return Comparator(
        name='equal',
        function=compare_equal,
        value=value1,
    )


def compare_less_than(value1, value2) -> bool:
    return value1 < value2


def LESS_THAN(value1) -> callable:
    return Comparator(
        name='lessthan',
        function=compare_less_than,
        value=value1,
    )


def compare_less_than_equals(value1, value2) -> bool:
    return value1 <= value2


def LESS_THAN_EQUALS(value1) -> Comparator:

    return Comparator(
        name='lessthanequals',
        function=compare_less_than_equals,
        value=value1,
    )


def compare_greater_than(value1, value2) -> bool:
    return value1 > value2


def GREATER_THAN(value1) -> Comparator:

    return Comparator(
        name='greaterthan',
        function=compare_greater_than,
        value=value1,
    )


def compare_greater_than_equals(value1, value2) -> bool:
    return value1 >= value2


def GREATER_THAN_EQUALS(value1) -> Comparator:
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


def ATOMS_EQUAL_TO(atoms1):

    return Comparator(
        name='atomsequalto',
        function=compare_atoms,
        value=atoms1,
    )


def compare_is(obj1, obj2):
    return obj1 is obj2


def IS(obj1) -> Comparator:

    return Comparator(
        name='atomsequalto',
        function=compare_is,
        value=obj1,
    )


def compare_contains(obj1, obj2):
    return obj1 in obj2


def CONTAINS(obj1):
    return Comparator(
        name='contains',
        function=compare_contains,
        value=obj1,
    )


def CALCULATORSPEC(obj):
    return Comparator(
        name='calculator_spec',
        function=compare_calculator,
        value=obj,
    )


def compare_calculator(calc1, calc2):
    name1 = calc1.get('name')
    name2 = calc2.get('name')
    if name1 != name2:
        return False

    if name1 == 'gpaw':
        return compare_equal(calc1, calc2)
    elif name1 == 'emt':
        return True
    else:
        raise NotImplementedError


comparators = types.SimpleNamespace(
    EQUAL=EQUAL,
    IS=IS,
    LESS_THAN=LESS_THAN,
    GREATER_THAN=GREATER_THAN,
    LESS_THAN_EQUALS=LESS_THAN_EQUALS,
    GREATER_THAN_EQUALS=GREATER_THAN_EQUALS,
    APPROX=APPROX,
    ATOMS_EQUAL_TO=ATOMS_EQUAL_TO,
    ANY=ANY,
    CONTAINS=CONTAINS,
    ALL=ALL,
    OR=OR,
    AND=AND,
    NOT=NOT,
    CALCULATORSPEC=CALCULATORSPEC,
)
