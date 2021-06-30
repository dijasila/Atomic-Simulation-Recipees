"""Implement cache functionality."""
import os
import pathlib
from pathlib import Path
import typing
from .record import Record
from .utils import write_file, only_master, link_file
from .serialize import JSONSerializer
from .selector import Selector
from .filetype import find_external_files, ASRPath
from .lock import Lock, lock


class DuplicateRecord(Exception):
    pass


def get_external_file_path(dir, uid, name):
    newpath = dir / ('-'.join([uid[:10], name]))
    return newpath


class FileCacheBackend:

    def __init__(self, path: Path):
        self.serializer = JSONSerializer()
        self.path = path
        self.cache_dir = ASRPath('records')
        self.ext_file_dir = ASRPath('external_files')
        self.record_table_path = ASRPath('record-table.json')
        self.lock = Lock(ASRPath('lock'), timeout=10)

    def _record_to_path(self, run_record: Record):
        run_specification = run_record.run_specification
        run_uid = run_specification.uid
        name = run_record.run_specification.name + '-' + run_uid[:10]
        return self.cache_dir / f'{name}.json'

    @lock
    def add(self, run_record: Record):
        self.initialize()
        run_specification = run_record.run_specification
        run_uid = run_specification.uid

        external_files = find_external_files(run_record.result)
        for external_file in external_files:
            asr_path = get_external_file_path(
                dir=self.ext_file_dir,
                uid=run_uid,
                name=external_file.name,
            )
            only_master(link_file)(external_file.path, asr_path)
            external_file.path = asr_path

        pth = self._record_to_path(run_record)
        serialized_object = self.serializer.serialize(run_record)

        self._write_file(pth, serialized_object)
        self.add_uid_to_table(run_uid, pth)
        return run_uid

    @lock
    def update(self, run_record: Record):
        self.initialize()

        # Basically the same as add but without touching side effects.
        run_specification = run_record.run_specification
        run_uid = run_specification.uid
        pth = self._record_to_path(run_record)
        serialized_object = self.serializer.serialize(run_record)

        self._write_file(pth, serialized_object)
        self.add_uid_to_table(run_uid, pth)
        return run_uid

    @property
    def initialized(self):
        from asr.core.root import Repository
        if not Repository.root_is_initialized():
            return False
        return self.record_table_path.is_file()

    @lock
    def initialize(self):
        if self.initialized:
            return

        if not self.cache_dir.is_dir():
            only_master(os.makedirs)(self.cache_dir)

        if not self.ext_file_dir.is_dir():
            only_master(os.makedirs)(self.ext_file_dir)

        serialized_object = self.serializer.serialize({})
        self._write_file(self.record_table_path, serialized_object)

    @lock
    def add_uid_to_table(self, run_uid, path: ASRPath):
        self.initialize()
        uid_table = self.uid_table
        uid_table[run_uid] = path
        self._write_file(
            self.record_table_path,
            self.serializer.serialize(uid_table),
        )

    @lock
    def remove_uid_from_table(self, run_uid):
        assert self.initialized
        uid_table = self.uid_table
        del uid_table[run_uid]
        self._write_file(
            self.record_table_path,
            self.serializer.serialize(uid_table),
        )

    @property
    def uid_table(self):
        self.initialize()
        text = self._read_file(self.record_table_path)
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

    def _immediate_dependencies(self, record):
        # XXX We should probably get a O(1) __contains__ method
        # XXX record == record2 seems to fail with an error
        assert record.uid == self.get_record_from_uid(record.uid).uid
        if not record.dependencies:
            return
        for dep in record.dependencies:
            yield self.get_record_from_uid(dep.uid)

    def _all_dependencies_with_duplicates(self, record):
        for dep_record in self._immediate_dependencies(record):
            yield dep_record
            yield from self._all_dependencies_with_duplicates(dep_record)

    def recurse_dependencies(self, record):
        found = {record.uid}
        for dep_record in self._all_dependencies_with_duplicates(record):
            uid = dep_record.uid
            if uid not in found:
                found.add(uid)
                yield dep_record

    def get_record_from_uid(self, run_uid):
        path = self.uid_table[run_uid]
        serialized_object = self._read_file(path)
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

    @lock
    def remove(self, selector: Selector = None):
        assert self.initialized, 'No cache here!'
        all_records = [self.get_record_from_uid(run_uid)
                       for run_uid in self.uid_table]
        if selector is None:
            return []

        selected = []
        for record in all_records:
            if selector.matches(record):
                selected.append(record)

        for record in selected:
            self.remove_uid_from_table(record.uid)
            pth = self._record_to_path(record)
            pth.unlink()
        return selected

    def _write_file(self, path: ASRPath, text: str):
        write_file(path, text)

    def _read_file(self, path: ASRPath) -> str:
        serialized_object = pathlib.Path(path).read_text()
        return serialized_object


class Cache:

    def __init__(self, backend):
        self.backend = backend

    @staticmethod
    def make_selector(selector: Selector = None, equals={}):
        if selector is None:
            selector = Selector()

        for key, value in equals.items():
            setattr(selector, key, selector.EQUAL(value))

        return selector

    def add(self, run_record: Record):
        selector = self.make_selector()
        selector.run_specification.uid = (
            selector.EQUAL(run_record.run_specification.uid)
        )

        self.backend.add(run_record)

    def update(self, record: Record):
        """Update existing record with record.uid."""
        selector = self.make_selector()
        selector.run_specification.uid = selector.EQUAL(record.uid)
        if hasattr(self.backend, 'update'):
            self.backend.update(record)
        else:
            self.backend.add(record)

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

    def remove(self, *, selector: Selector = None, **equals):
        """Remove records."""
        selector = self.make_selector(selector=selector, equals=equals)
        return self.backend.remove(selector)

    def __call__(self, make_selector=None):
        if make_selector is None:
            make_selector = default_make_selector

        def wrapper(func):
            def wrapped(run_specification):
                sel = make_selector(run_specification)
                if self.has(selector=sel):
                    run_record = self.get(selector=sel)
                    print(f'{run_specification.name}: '
                          f'Found cached record.uid={run_record.uid}')
                else:
                    run_record = func(run_specification)
                    self.add(run_record)

                return run_record
            return wrapped

        return wrapper

    def __contains__(self, record: Record):
        return self.has(uid=record.uid)


def default_make_selector(run_specification):
    selector = Selector()
    selector.run_specification.name = selector.EQ(run_specification.name)
    selector.run_specification.parameters = \
        selector.EQ(run_specification.parameters)
    selector.run_specification.version = selector.EQ(run_specification.version)
    return selector


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

    def remove(self, selector: Selector):
        selected = []
        for record in self.records.values():
            if selector.matches(record):
                selected.append(record)

        for record in selected:
            del self.records[record.uid]
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
        from asr.core.root import Repository
        repo = Repository.find_root()
        return repo.cache
    elif backend == 'memory':
        return Cache(backend=MemoryBackend())
