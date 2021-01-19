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


def find_root(path: str = '.'):
    path = pathlib.Path(path).absolute()
    if (path / config.root).is_dir():
        return path
    abspath = str(path)
    strroot = str(config.root)
    assert strroot in abspath
    return pathlib.Path(abspath.split(strroot)[0])
