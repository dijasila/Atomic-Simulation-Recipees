"""Implement RunSpec and Record."""
from __future__ import annotations
import numpy as np

import typing
import copy
from dataclasses import dataclass
from .specification import RunSpecification
from .resources import Resources
from .metadata import Metadata
from .history import History

# XXX: Make Tags object.


@dataclass
class Record:

    result: typing.Optional[typing.Any] = None
    run_specification: typing.Optional[RunSpecification] = None
    resources: typing.Optional[Resources] = None
    dependencies: typing.Optional[typing.List[str]] = None
    history: typing.Optional[History] = None
    tags: typing.Optional[typing.List[str]] = None
    metadata: typing.Optional[Metadata] = None

    @property
    def parameters(self):
        return self.run_specification.parameters

    @property
    def uid(self):
        return self.run_specification.uid

    @property
    def version(self):
        return self.run_specification.version

    @version.setter
    def version(self, value):
        self.run_specification.version = value

    @property
    def name(self):
        return self.run_specification.name

    def copy(self):
        data = copy.deepcopy(self.__dict__)
        return Record(**data)

    def __str__(self):
        strings = []
        for name, value in self.__dict__.items():
            if name == 'result':
                txt = str(value)
                if len(txt) > 30:
                    strings.append('result=' + str(value)[:30] + '...')
                    continue
            if value is not None:
                strings.append('='.join([str(name), str(value)]))
        return 'Record(' + ', '.join(strings) + ')'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Record):
            return False

        return compare_dct_with_numpy_arrays(self.__dict__, other.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]


def compare_dct_with_numpy_arrays(dct1, dct2):
    """Compare dictionaries that might containt a numpy array.

    Numpy array are special since their equality test can return an
    array and meaning that we cannot just evaluate dct1 == dct2 since
    that would raise an error.

    """
    dct1keys = dct1.keys()
    dct2keys = dct2.keys()
    for key in dct1keys:
        if key not in dct2keys:
            return False
        value1 = dct1[key]
        value2 = dct2[key]

        if isinstance(value1, np.ndarray):
            if not np.array_equal(value1, value2):
                return False
        else:
            if not value1 == value2:
                return False
    return True
