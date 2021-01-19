import pathlib
from asr.core.utils import sha256sum
from .config import find_root, relative_to_root


class ExternalFile:

    def __init__(self, path: pathlib.Path):
        self.path = str(path.absolute())
        self.hashes = {'sha256': sha256sum(path)}

    @classmethod
    def fromstr(cls, string):
        return cls(pathlib.Path(string))

    @property
    def sha256(self):
        return self.hashes['sha256']

    @property
    def name(self):
        return pathlib.Path(self.path).name

    def __str__(self):
        return f'ExternalFile(path={self.path}, sha256={self.sha256[:10]}...)'

    def __repr__(self):
        return str(self)

    def __fspath__(self):
        return str(self.path)

    def restore(self):
        path = self.path
        tofile = pathlib.Path(self.filename)
        assert not tofile.is_file()
        tofile.write_bytes(path.read_bytes())

    def __eq__(self, other):
        if not isinstance(other, ExternalFile):
            return NotImplemented
        return self.sha256 == other.sha256


class ASRFile(ExternalFile):
    """File relative to the .asr root folder."""

    def __init__(self, path: pathlib.Path):
        self.path = str(relative_to_root(path.absolute()))
        self.hashes = {'sha256': sha256sum(path)}

    def __str__(self):
        return f'ASRFile(path={self.path}, sha256={self.sha256[:10]}...)'

    def __fspath__(self):
        return str(find_root() / self.path)

    def __eq__(self, other):
        if not isinstance(other, ASRFile):
            return NotImplemented
        return self.sha256 == other.sha256
