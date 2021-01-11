"""Implement cache functionality."""
import os
import pathlib
import fnmatch
import uuid
from .record import RunRecord
from .specification import RunSpecification
from .utils import write_file, only_master
from .serialize import Serializer, JSONSerializer
from .selection import Selection


class RunSpecificationAlreadyExists(Exception):
    pass


class LegacyFileSystemBackend:

    def migrate(self):
        pass

    def add(self, run_record: RunRecord):
        raise NotImplementedError

    @property
    def uid_table(self):
        table = {}
        for pattern in self.relevant_patterns:
            for path in pathlib.Path().glob(pattern):
                filepath = str(path)
                if any(fnmatch.fnmatch(filepath, skip_pattern)
                       for skip_pattern in self.skip_patterns):
                    continue
                table[str(path)] = str(path)
        return table

    def has(self, selection: 'Selection'):
        records = self.select()
        for record in records:
            if selection.matches(record):
                return True
        return False

    def get_result_from_uid(self, uid):
        from asr.core import read_json
        filename = self.uid_table[uid]
        result = read_json(filename)
        return result

    def get_record_from_uid(self, uid):
        from asr.core.results import MetaDataNotSetError
        result = self.get_result_from_uid(uid)
        try:
            parameters = result.metadata.params
        except MetaDataNotSetError:
            parameters = {}
        try:
            code_versions = result.metadata.code_versions
        except MetaDataNotSetError:
            code_versions = {}

        name = result.metadata.asr_name

        if '@' not in name:
            name += '::main'
        else:
            name = name.replace('@', '::')

        return RunRecord(
            run_specification=RunSpecification(
                name=name,
                parameters=parameters,
                version=-1,
                codes=code_versions,
                uid=uuid.uuid4().hex,
            ),
            result=result,
        )

    def select(self, selection: 'Selection' = None):
        all_records = [self.get_record_from_uid(uid)
                       for uid in self.uid_table]
        if selection is None:
            return all_records
        selected = []
        for record in all_records:
            if selection.matches(record):
                selected.append(record)
        return selected


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
        serialized_object = self.serializer.serialize({})
        self._write_file(self.filename, serialized_object)

    def add_uid_to_table(self, run_uid, filename):
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

    def has(self, selection: 'Selection'):
        if not self.initialized:
            return False
        records = self.select()
        for record in records:
            if selection.matches(record):
                return True
        return False

    def get_record_from_uid(self, run_uid):
        filename = self.uid_table[run_uid]
        serialized_object = self._read_file(filename)
        obj = self.serializer.deserialize(serialized_object)
        return obj

    def select(self, selection: 'Selection' = None):
        if not self.initialized:
            return []
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
            only_master(os.makedirs)(self.cache_dir)
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

    def add(self, run_record: RunRecord):
        selection = {'run_specification.uid': run_record.run_specification.uid}
        has_uid = self.has(**selection)
        assert not has_uid, (
            'This uid already exists in the cache. Cannot overwrite.'
        )
        self.backend.add(run_record)

    def update(self, record: RunRecord):
        """Update existing record with record.uid."""
        selection = {'run_specification.uid': record.uid}
        has_uid = self.has(**selection)
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

    def has(self, **selection):
        selection = Selection(
            **selection,
        )
        return self.backend.has(selection)

    def get(self, **selection):
        assert self.has(**selection), \
            'No matching run_specification.'
        records = self.select(**selection)
        assert len(records) == 1, \
            f'More than one record matched! records={records}'
        return records[0]

    def select(self, **selection):
        """Select records.

        Selection can be in the style of

        cache.select(uid=uid)
        cache.select(name='asr.gs::main')
        """
        selection = Selection(
            **selection
        )
        return self.backend.select(selection)

    def wrapper(self, func):
        def wrapped(asrcontrol, run_specification):
            selection = {
                'run_specification.name': run_specification.name,
                'run_specification.parameters': run_specification.parameters,
                'migrated_to': lambda migrated_to: migrated_to is None,
            }
            if self.has(**selection):
                run_record = self.get(**selection)
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

    def has(self, selection: Selection):
        for value in self.records.values():
            if selection.matches(value):
                return True
        return False

    def select(self, selection: Selection):
        selected = []
        for record in self.records.values():
            if selection.matches(record):
                selected.append(record)
        return selected


def get_cache():
    from .config import config

    if config.backend == 'fscache':
        return file_system_cache
    elif config.backend == 'legacyfscache':
        return legacy_file_system_cache


file_system_cache = Cache(backend=FileCacheBackend())
legacy_file_system_cache = Cache(backend=LegacyFileSystemBackend())
