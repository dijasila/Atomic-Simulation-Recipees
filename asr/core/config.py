import pathlib
from asr.core import read_json


backends = {'memory', 'filesystem'}


class Config:
    defaults = {
        'backend': 'filesystem',
        'root': '.asr'
    }

    types = {
        'backend': str,
        'root': pathlib.Path,
    }

    @property
    def data(self):
        try:
            return read_json('asrconfig.json')
        except FileNotFoundError:
            return {}

    def __getattr__(self, attr):
        assert attr in self.defaults, f'Unknown attr={attr}'
        attribute = self.data.get(attr, self.defaults[attr])
        tp = self.types[attr]
        return tp(attribute)


config = Config()


def relative_to_root(path):
    parts = path.absolute().parts
    root = str(config.root)
    count = parts.count(root)
    assert count == 1, f'path={path} must contain {root} exactly once!'
    ind = parts.index(root)
    return pathlib.Path().joinpath(*parts[ind:])


class ASRRootNotFound(Exception):
    pass


def root_is_initialized():
    try:
        find_root()
        return True
    except ASRRootNotFound:
        return False


def initialize_root():
    if not root_is_initialized():
        config.root.mkdir()


def find_root(path: str = '.'):
    path = pathlib.Path(path).absolute()
    strroot = str(config.root)
    if (path / config.root).is_dir():
        return path / strroot
    abspath = str(path)
    if strroot not in abspath:
        raise ASRRootNotFound
    return pathlib.Path(abspath.split(strroot)[0]) / strroot
