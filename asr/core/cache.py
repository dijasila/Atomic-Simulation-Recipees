"""Implement cache functionality."""
import os
import pathlib
import typing
from .record import RunRecord
from .utils import write_file, only_master
from .serialize import Serializer, JSONSerializer
from .selector import Selector


class DuplicateRecord(Exception):
    pass


class FileCacheBackend():

    def __init__(
            self,
            cache_dir: str = '.asr/records',
            serializer: Serializer = JSONSerializer(),
    ):
        self.serializer = serializer
        self.cache_dir = pathlib.Path(cache_dir)
        self.filename = 'run-data.json'

    @staticmethod
    def _name_to_results_filename(name: str):
        return f'results-{name}.json'

    def add(self, run_record: RunRecord):
        if not self.initialized:
            self.initialize()
        run_specification = run_record.run_specification
        run_uid = run_specification.uid
        name = run_record.run_specification.name + '-' + run_uid[:10]
        filename = self._name_to_results_filename(name)
        serialized_object = self.serializer.serialize(run_record)
        self._write_file(filename, serialized_object)
        self.add_uid_to_table(run_uid, filename)
        return run_uid

    @property
    def initialized(self):
        return (self.cache_dir / pathlib.Path(self.filename)).is_file()

    def initialize(self):
        assert not self.initialized

        if not self.cache_dir.is_dir():
            only_master(os.makedirs)(self.cache_dir)
        serialized_object = self.serializer.serialize({})
        self._write_file(self.filename, serialized_object)

    def add_uid_to_table(self, run_uid, filename):
        if not self.initialized:
            self.initialize()
        uid_table = self.uid_table
        uid_table[run_uid] = filename
        self._write_file(
            self.filename,
            self.serializer.serialize(uid_table)
        )

    @property
    def uid_table(self):
        if not self.initialized:
            self.initialize()
        text = self._read_file(self.filename)
        uid_table = self.serializer.deserialize(text)
        return uid_table

    def has(self, selector: 'Selector'):
        if not self.initialized:
            return False
        records = self.select()
        for record in records:
            if selector.matches(record):
                return True
        return False

    def get_record_from_uid(self, run_uid):
        filename = self.uid_table[run_uid]
        serialized_object = self._read_file(filename)
        obj = self.serializer.deserialize(serialized_object)
        return obj

    def select(self, selector: Selector = None):
        if not self.initialized:
            return []
        all_records = [self.get_record_from_uid(run_uid)
                       for run_uid in self.uid_table]
        if selector is None:
            return all_records
        selected = []
        for record in all_records:
            if selector.matches(record):
                selected.append(record)
        return selected

    def _write_file(self, filename: str, text: str):
        write_file(self.cache_dir / filename, text)

    def _read_file(self, filename: str) -> str:
        serialized_object = pathlib.Path(self.cache_dir / filename).read_text()
        return serialized_object


class Cache:

    def get_migrations(self):
        """Migrate cache data."""
        from asr.core.migrate import (
            Migrations,
            generate_resultsfile_migrations,
            generate_record_migrations,
        )
        migrations = Migrations(
            generators=[
                generate_resultsfile_migrations,
                generate_record_migrations,
            ],
            cache=self,
        )
        return migrations

    def __init__(self, backend):
        self.backend = backend

    @staticmethod
    def make_selector(selector: Selector = None, equals={}):
        if selector is None:
            selector = Selector()

        for key, value in equals.items():
            setattr(selector, key, selector.EQUAL(value))

        return selector

    def add(self, run_record: RunRecord):
        selector = self.make_selector()
        selector.run_specification.uid = (
            selector.EQUAL(run_record.run_specification.uid)
        )

        has_uid = self.has(selector=selector)
        assert not has_uid, (
            'This uid already exists in the cache. Cannot overwrite.'
        )
        self.backend.add(run_record)

    def update(self, record: RunRecord):
        """Update existing record with record.uid."""
        selector = self.make_selector()
        selector.run_specification.uid = selector.EQUAL(record.uid)
        has_uid = self.has(selector=selector)
        assert has_uid, 'Unknown run UID to update.'
        self.backend.add(record)

    def migrate_record(
            self, original_record, migrated_record, migration_label):
        from asr.core.specification import get_new_uuid
        migrated_uid = get_new_uuid()
        original_uid = original_record.uid

        migrated_record.run_specification.uid = migrated_uid

        original_record.migrated_to = migrated_uid
        migrated_record.migrated_from = original_uid
        migrated_record.migrations.append(migration_label)
        self.update(original_record)
        self.add(migrated_record)

    def has(self, *, selector: Selector = None, **equals):
        selector = self.make_selector(selector, equals)
        return self.backend.has(selector)

    def get(self, *, selector: Selector = None, **equals):
        selector = self.make_selector(selector, equals)
        records = self.select(selector=selector)
        assert records, 'No matching run_specification.'
        assert len(records) == 1, \
            f'More than one record matched! records={records}'
        return records[0]

    def select(self, *, selector: Selector = None, **equals):
        """Select records.

        Selector can be in the style of

        cache.select(uid=uid)
        cache.select(name='asr.gs::main')
        """
        selector = self.make_selector(selector=selector, equals=equals)
        return self.backend.select(selector)

    def wrapper(self, func):
        def wrapped(asrcontrol, run_specification):

            equals = {
                'run_specification.name': run_specification.name,
                'run_specification.parameters': run_specification.parameters,
            }
            sel = self.make_selector(equals=equals)
            sel.migrated_to = sel.IS(None)

            if self.has(selector=sel):
                run_record = self.get(selector=sel)
                print(f'{run_specification.name}: '
                      f'Found cached record.uid={run_record.uid}')
            else:
                run_record = func(asrcontrol, run_specification)
                self.add(run_record)

            return run_record
        return wrapped

    def __call__(self):
        return self.wrapper

    def __contains__(self, record):
        return self.has(uid=record.uid)


class MemoryBackend:

    def __init__(self):
        self.records = {}

    def add(self, record):
        self.records[record.uid] = record

    def has(self, selector: Selector):
        for value in self.records.values():
            if selector.matches(value):
                return True
        return False

    def select(self, selector: Selector):
        selected = []
        for record in self.records.values():
            if selector.matches(record):
                selected.append(record)
        return selected


def get_cache(backend: typing.Optional[str] = None) -> Cache:
    """Get ASR Cache object.

    Parameters
    ----------
    backend
        The chosen backend. Allowed values 'filesystem', 'memory'.
    """
    if backend is None:
        from .config import config
        backend = config.backend

    if backend == 'filesystem':
        return Cache(backend=FileCacheBackend())
    elif backend == 'memory':
        return Cache(backend=MemoryBackend())

