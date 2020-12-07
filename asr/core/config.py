from asr.core import read_json


backends = {'legacyfscache', 'fscache'}


class Config:

    @property
    def data(self):
        try:
            return read_json('asrconfig.json')
        except FileNotFoundError:
            return {}

    @property
    def backend(self):
        backend = self.data.get('backend', 'fscache')
        assert backend in backends
        return backend


config = Config()
