import pathlib
from asr.core import read_json
from .root import (  # noqa
    root_is_initialized, initialize_root, find_root
)


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
