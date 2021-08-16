from os import PathLike
import typing
from pathlib import Path
from asr.core.utils import sha256sum
from .utils import only_master, link_file


class ExternalFile:
    def __init__(self, path: PathLike):
        self.path = Path(path).absolute()
        self.hashes = {'sha256': sha256sum(path)}

    @property
    def name(self):
        return self.path.name

    @classmethod
    def fromstr(cls, string):
        path = Path(string).absolute()
        return cls(path)

    @property
    def sha256(self):
        return self.hashes['sha256']

    def __repr__(self):
        return f'ExternalFile(path={self.path}, sha256={self.sha256[:10]}...)'

    def __fspath__(self):
        return str(self.path)

    def restore(self):
        path = self.path
        tofile = Path(path.name)
        assert not tofile.is_file()
        only_master(link_file)(path, tofile)

    def __eq__(self, other):
        if not isinstance(other, ExternalFile):
            return NotImplemented
        return self.sha256 == other.sha256


class File:

    def __init__(
            self,
            path: Path,
    ):
        self.path = path
        self.hashes = {'sha256': sha256sum(path)}

    @classmethod
    def fromstr(cls, string):
        path = Path(string).absolute()
        return cls(path)

    @property
    def sha256(self):
        return self.hashes['sha256']

    def __repr__(self):
        return f'File(path={self.path}, sha256={self.sha256[:10]}...)'

    def __fspath__(self):
        return str(self.path)

    def __eq__(self, other):
        if not isinstance(other, File):
            return NotImplemented
        return self.sha256 == other.sha256


def is_external_file(obj):
    return isinstance(obj, ExternalFile)


def find_external_files(obj) -> typing.List[ExternalFile]:

    if is_external_file(obj):
        return [obj]

    if isinstance(obj, (list, tuple)):
        items = obj
    elif isinstance(obj, dict):
        items = obj.values()
    elif hasattr(obj, '__dict__'):
        items = obj.__dict__.values()
    else:
        items = []
    external_files = []
    for item in items:
        external_files.extend(find_external_files(item))
    return external_files
