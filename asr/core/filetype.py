import typing
import pathlib
from asr.core.utils import sha256sum
from .utils import only_master, link_file
from .config import find_root


class ASRPath:
    """Pathlike object relative to ASR root."""

    def __init__(self, path: pathlib.Path):
        assert not path.is_absolute()
        self.path = path

    def __fspath__(self):
        return str(find_root() / self.path)

    def __str__(self):
        return self.__fspath__()

    def __eq__(self, other):
        if not isinstance(other, ASRPath):
            return False
        return self.path == other.path

    def __repr__(self):
        return str(self)


PathLike = typing.Union[pathlib.Path, ASRPath]


class ExternalFile:

    def __init__(
            self,
            path: PathLike,
            name: str,
    ):
        self.path = path
        self.name = name
        self.hashes = {'sha256': sha256sum(path)}

    @classmethod
    def fromstr(cls, string):
        path = pathlib.Path(string).absolute()
        return cls(path, path.name)

    @property
    def sha256(self):
        return self.hashes['sha256']

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    def __str__(self):
        return f'ExternalFile(path={self.path}, sha256={self.sha256[:10]}...)'

    def __repr__(self):
        return str(self)

    def __fspath__(self):
        return str(self.path)

    def restore(self):
        path = self.path
        tofile = pathlib.Path(self.name)
        assert not tofile.is_file()
        only_master(link_file)(path, tofile)

    def __eq__(self, other):
        if not isinstance(other, ExternalFile):
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
