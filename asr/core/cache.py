"""Implement cache functionality."""
import abc
import functools
import os
import pathlib
import hashlib
from .record import RunRecord
from .specification import RunSpecification, construct_run_spec
from .utils import write_file
from .serialize import Serializer, JSONSerializer


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


class FullFeatureFileCache(AbstractCache):  # noqa

    def __init__(self, serializer: Serializer = JSONSerializer(),  # noqa
                 hash_func=hashlib.sha256):  # noqa
        self.serializer = serializer
        self._cache_dir = pathlib.Path('.asr/records')
        self.depth = 0
        self.hash_func = hash_func
        self._filename = 'run-data.json'

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
        run_specification = run_record.run_specification
        if self.has(run_record.run_specification):
            raise RunSpecificationAlreadyExists(
                'This Run specification already exists in cache.'
            )

        run_hash = self.get_hash(run_specification)
        name = run_record.run_specification.name + '-' + run_hash[:10]
        filename = self._name_to_results_filename(name)
        serialized_object = self.serializer.serialize(run_record)
        self._write_file(filename, serialized_object)
        self.add_hash_to_table(run_hash, filename)
        return run_hash

    def get_hash(self, run_specification: RunSpecification):  # noqa
        run_spec_to_be_hashed = construct_run_spec(  # noqa
            name=run_specification.name,
            parameters=run_specification.parameters,
            version=run_specification.version,
            uid='0',
        )
        serialized_object = self.serializer.serialize(run_spec_to_be_hashed)
        return self.hash_func(serialized_object.encode()).hexdigest()

    @property
    def initialized(self):  # noqa
        return (self.cache_dir / pathlib.Path(self._filename)).is_file()

    def initialize(self):  # noqa
        assert not self.initialized
        serialized_object = self.serializer.serialize({})
        self._write_file(self._filename, serialized_object)

    def add_hash_to_table(self, run_hash, filename):  # noqa
        hash_table = self.hash_table
        hash_table[run_hash] = filename
        self._write_file(
            self._filename,
            self.serializer.serialize(hash_table)
        )

    @property
    def hash_table(self):  # noqa
        if not self.initialized:
            self.initialize()
        text = self._read_file(self._filename)
        hash_table = self.serializer.deserialize(text)
        return hash_table

    @property
    def hashes(self):  # noqa
        return self.hash_table.keys()

    def has(self, run_specification: RunSpecification):  # noqa
        run_hash = self.get_hash(run_specification)
        return run_hash in self.hashes

    def get(self, run_specification: RunSpecification):  # noqa
        assert self.has(run_specification), \
            'No matching run_specification.'
        run_hash = self.get_hash(run_specification)
        return self.get_record_from_hash(run_hash)

    def get_record_from_hash(self, run_hash):  # noqa
        filename = self.hash_table[run_hash]
        serialized_object = self._read_file(filename)
        obj = self.serializer.deserialize(serialized_object)
        return obj

    def get_record_from_uid(self, uid):  # noqa
        return [record for record in self.select() if record.uid == uid][0]

    def select(self):  # noqa
        return [self.get_record_from_hash(run_hash)
                for run_hash in self.hash_table]

    def _write_file(self, filename: str, text: str):
        if not self.cache_dir.is_dir():
            os.makedirs(self.cache_dir)
        write_file(self.cache_dir / filename, text)

    def _read_file(self, filename: str) -> str:
        serialized_object = pathlib.Path(self.cache_dir / filename).read_text()
        return serialized_object

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, type, value, traceback):
        """Exit context manager."""
        pass

    def wrapper(self, func):  # noqa
        def wrapped(asrcontrol, run_specification):
            with self:
                if self.has(run_specification):
                    run_record = self.get(run_specification)
                    print(f'Using cached record: {run_record}')
                else:
                    run_record = func(asrcontrol, run_specification)
                    self.add(run_record)
            return run_record
        return wrapped

    def __call__(self):  # noqa
        return functools.partial(self.wrapper)


single_run_file_cache = SingleRunFileCache()
full_feature_file_cache = FullFeatureFileCache()
