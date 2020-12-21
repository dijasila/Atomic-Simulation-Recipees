import pathlib
from asr.core.utils import sha256sum
from .config import relative_to_root, find_root


class ExternalFile:  # noqa

    def __init__(self, path: pathlib.Path):  # noqa
        self.path = str(relative_to_root(path.absolute()))
        self.filename = str(path)
        self.hashes = {'sha256': sha256sum(path)}

    def __str__(self):  # noqa
        return f'ExternalFile({self.path})'

    def __repr__(self):
        return str(self)

    def __fspath__(self):  # noqa
        return str(find_root() / self.path)

    def checksum(self, id='sha256'):
        return self.hashes[id]

    def restore(self):  # noqa
        assert not self.filename == self.path, \
            'The file already exists.'
        path = find_root() / pathlib.Path(self.path)
        pathlib.Path(self.filename).write_bytes(
            path.read_bytes())
