from pathlib import Path


error_not_initialized = """\
Root directory not initialized in working directory \
"{directory}" or any of its parents.  Please run "asr init" \
in a suitable directory to initialize the root directory"""


class ASRRootNotFound(Exception):
    pass


class Repository:
    def __init__(self, root: Path):
        from asr.core.cache import Cache, FileCacheBackend

        root = root.absolute()
        cache_dir = root / ASR_DIR

        if not cache_dir.is_dir():
            raise ASRRootNotFound(f'Root not initialized in {self.root}')

        self.cache = Cache(backend=FileCacheBackend(cache_dir))

    @property
    def root(self):
        return self.cache.path.parent

    @classmethod
    def find_root(cls, path='.'):
        root = find_root(path)
        return cls(root)

    @classmethod
    def root_is_initialized(cls):
        try:
            cls.find_root()
        except ASRRootNotFound:
            return False
        return True

    @classmethod
    def initialize(cls, root: Path):
        initialize_root(root)
        return cls(root)

    def __repr__(self):
        return f'{type(self).__name__}(root={self.root})'




def initialize_root(directory: Path = Path('.')):
    asr_dir = directory / ASR_DIR
    assert not asr_dir.exists()
    asr_dir.mkdir()


def find_root(path: str = '.'):
    path = Path(path).absolute()
    origpath = path
    while not (path / ASR_DIR).is_dir():
        if path == Path('/'):
            raise ASRRootNotFound(error_not_initialized
                                  .format(directory=origpath))
        path = path.parent
    assert (path / ASR_DIR).is_dir()
    return path


def find_repo_root(path: str = '.'):
    root = find_root()
    return root / ASR_DIR


ASR_DIR = Path('.asr')
