"""Implement cache functionality."""
import abc
import os
import pathlib
import functools
import numpy as np
from .record import RunRecord
from .specification import RunSpecification
from .utils import write_file
from .serialize import Serializer, JSONSerializer
from ase import Atoms


class RunSpecificationAlreadyExists(Exception):  # noqa
    pass


class AbstractCache(abc.ABC):
    """Abstract cache interface."""

    @abc.abstractmethod
    def add(self, run_record: RunRecord):  # noqa
        pass

    @abc.abstractmethod
    def get(self, run_record: RunRecord):  # noqa
        pass

    @abc.abstractmethod
    def has(self, run_specification: RunSpecification):  # noqa
        pass


class NoCache(AbstractCache):  # noqa

    def add(self, run_record: RunRecord):
        """Add record of run to cache."""
        ...

    def has(self, run_specification: RunSpecification) -> bool:
        """Has run record matching run specification."""
        return False

    def get(self, run_specification: RunSpecification):
        """Get run record matching run specification."""
        ...


class SingleRunFileCache(AbstractCache):  # noqa

    def __init__(self, serializer: Serializer = JSONSerializer()):  # noqa
        self.serializer = serializer
        self._cache_dir = None
        self.depth = 0

    @property
    def cache_dir(self) -> pathlib.Path:  # noqa
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value):
        self._cache_dir = value

    @staticmethod
    def _name_to_results_filename(name: str):
        name = name.replace('::', '@').replace('@main', '')
        return f'results-{name}.json'

    def add(self, run_record: RunRecord):  # noqa
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
        return filename

    def has(self, run_specification: RunSpecification):  # noqa
        name = run_specification.name
        filename = self._name_to_results_filename(name)
        return (self.cache_dir / filename).is_file()

    def get(self, run_specification: RunSpecification):  # noqa
        name = run_specification.name
        filename = self._name_to_results_filename(name)
        serialized_object = self._read_file(filename)
        obj = self.serializer.deserialize(serialized_object)
        return obj

    def select(self):  # noqa
        pattern = self._name_to_results_filename('*')
        paths = list(pathlib.Path(self.cache_dir).glob(pattern))
        serialized_objects = [self._read_file(path) for path in paths]
        deserialized_objects = [self.serializer.deserialize(ser_obj)
                                for ser_obj in serialized_objects]
        return deserialized_objects

    def _write_file(self, filename: str, text: str):
        write_file(self.cache_dir / filename, text)

    def _read_file(self, filename: str) -> str:
        serialized_object = pathlib.Path(self.cache_dir / filename).read_text()
        return serialized_object

    def __enter__(self):
        """Enter context manager."""
        if self.depth == 0:
            self.cache_dir = pathlib.Path('.').absolute()
        self.depth += 1
        return self

    def __exit__(self, type, value, traceback):
        """Exit context manager."""
        self.depth -= 1
        if self.depth == 0:
            self.cache_dir = None

    def __call__(self):  # noqa

        def wrapper(func):
            def wrapped(run_specification):
                with self:
                    if self.has(run_specification):
                        run_record = self.get(run_specification)
                    else:
                        run_record = func(run_specification)
                        self.add(run_record)
                return run_record
            return wrapped
        return wrapper


class FileCacheBackend:  # noqa

    def __init__(
            self,
            cache_dir: str = '.asr/records',
            serializer: Serializer = JSONSerializer(),
            # hash_func=hashlib.sha256,
    ):
        self.serializer = serializer
        self.cache_dir = pathlib.Path(cache_dir)
        # self.hash_func = hash_func
        self.filename = 'run-data.json'

    @staticmethod
    def _name_to_results_filename(name: str):
        return f'results-{name}.json'

    def add(self, run_record: RunRecord):  # noqa
        run_specification = run_record.run_specification
        run_uid = run_specification.uid
        # run_hash = self.get_hash(run_specification)
        name = run_record.run_specification.name + '-' + run_uid[:10]
        filename = self._name_to_results_filename(name)
        serialized_object = self.serializer.serialize(run_record)
        self._write_file(filename, serialized_object)
        self.add_uid_to_table(run_uid, filename)
        return run_uid

    # def get_hash(self, run_specification: RunSpecification):  # noqa
    #     run_spec_to_be_hashed = construct_run_spec(  # noqa
    #         name=run_specification.name,
    #         parameters=run_specification.parameters,
    #         version=run_specification.version,
    #         uid='0',
    #     )
    #     serialized_object = self.serializer.serialize(run_spec_to_be_hashed)
    #     return self.hash_func(serialized_object.encode()).hexdigest()

    @property
    def initialized(self):  # noqa
        return (self.cache_dir / pathlib.Path(self.filename)).is_file()

    def initialize(self):  # noqa
        assert not self.initialized
        serialized_object = self.serializer.serialize({})
        self._write_file(self.filename, serialized_object)

    def add_uid_to_table(self, run_uid, filename):  # noqa
        uid_table = self.uid_table
        uid_table[run_uid] = filename
        self._write_file(
            self.filename,
            self.serializer.serialize(uid_table)
        )

    @property
    def uid_table(self):  # noqa
        if not self.initialized:
            self.initialize()
        text = self._read_file(self.filename)
        uid_table = self.serializer.deserialize(text)
        return uid_table

    # @property
    # def hashes(self):  # noqa
    #     return self.hash_table.keys()

    def has(self, selection: 'Selection'):  # noqa
        records = self.select()
        for record in records:
            if selection.matches(record):
                return True
        return False
        # run_hash = self.get_hash(run_specification)
        # return run_hash in self.hashes

    # def get(self, run_specification: RunSpecification):  # noqa
    #     assert self.has(run_specification), \
    #         'No matching run_specification.'
    #     run_hash = self.get_hash(run_specification)
    #     return self.get_record_from_hash(run_hash)

    def get_record_from_uid(self, run_uid):  # noqa
        filename = self.uid_table[run_uid]
        serialized_object = self._read_file(filename)
        obj = self.serializer.deserialize(serialized_object)
        return obj

    # def get_record_from_uid(self, uid):  # noqa
    #     self.uid_table
    #     return
    # [record for record in self.select() if record.uid == uid][0]

    def select(self, selection: 'Selection' = None):
        all_records = [self.get_record_from_uid(run_uid)
                       for run_uid in self.uid_table]
        if selection is None:
            return all_records
        selected = []
        for record in all_records:
            if selection.matches(record):
                selected.append(record)
        return selected

    def _write_file(self, filename: str, text: str):
        if not self.cache_dir.is_dir():
            os.makedirs(self.cache_dir)
        write_file(self.cache_dir / filename, text)

    def _read_file(self, filename: str) -> str:
        serialized_object = pathlib.Path(self.cache_dir / filename).read_text()
        return serialized_object


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
    for value1, value2 in zip(flatlst1, flatlst2):
        if not value1 == value2:
            return False
    return True


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

    # def normalize_value(self, value):
    #     if isinstance(value, np.ndarray):
    #         return value.tolist()
    #     return value

    def normalize_selection(self, selection: dict):
        normalized = {}

        for key, value in selection.items():
            # value = self.normalize_value(value)
            comparator = None
            if isinstance(value, dict):
                norm = self.normalize_selection(value)
                for keynorm, valuenorm in norm.items():
                    normalized['.'.join([key, keynorm])] = valuenorm
            # elif isinstance(value, Parameters):
            #     norm = self.normalize_selection(value.__dict__)
            #     for keynorm, valuenorm in norm.items():
            #         normalized['.'.join([key, keynorm])] = valuenorm
            elif value is None:
                pass
            elif isinstance(value, Atoms):
                comparator = functools.partial(compare_atoms, value)
            elif isinstance(value, list):
                comparator = functools.partial(
                    compare_lists,
                    value,
                )
            elif type(value) in {str, bool, int}:
                comparator = value.__eq__
            elif type(value) is float:
                comparator = approx(value)
            elif hasattr(value, '__dict__'):
                norm = self.normalize_selection(value.__dict__)
                for keynorm, valuenorm in norm.items():
                    normalized['.'.join([key, keynorm])] = valuenorm
            else:
                raise AssertionError('Unknown type', type(value))

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

    def __str__(self):  # noqa
        return str(self.selection)


class Cache:  # noqa

    def __init__(self, backend):
        self.backend = backend

    def add(self, run_record: RunRecord):  # noqa
        if self.has(run_specification=run_record.run_specification):
            raise RunSpecificationAlreadyExists(
                'This Run specification already exists in cache.'
            )

        self.backend.add(run_record)

    def has(self, **selection):  # noqa
        selection = Selection(
            **selection,
        )
        return self.backend.has(selection)

    def get(self, **selection):  # noqa
        assert self.has(**selection), \
            'No matching run_specification.'
        records = self.select(**selection)
        assert len(records) == 1, 'More than one record matched!'
        return records[0]

    def select(self, **selection):  # noqa
        """Select records.

        Selection can be in the style of

        cache.select(uid=uid)
        cache.select(name='asr.gs::main')
        """
        selection = Selection(
            **selection
        )
        return self.backend.select(selection)

    def wrapper(self, func):  # noqa
        def wrapped(asrcontrol, run_specification):
            # run_spec_to_match = construct_run_spec(  # noqa
            #     name=run_specification.name,
            #     # parameters=run_specification.parameters,
            #     # version=run_specification.version,
            # )
            selection = {
                'run_specification.name': run_specification.name,
                'run_specification.parameters': run_specification.parameters,
            }
            if self.has(**selection):
                run_record = self.get(**selection)
                print(f'Using cached record: {run_record}')
            else:
                run_record = func(asrcontrol, run_specification)
                self.add(run_record)
            return run_record
        return wrapped

    def __call__(self):  # noqa
        return self.wrapper


file_system_cache = Cache(backend=FileCacheBackend())
