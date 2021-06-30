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

        self.root = root
        self.cache = Cache(backend=FileCacheBackend(cache_dir))

    def asr_path(self, path):
        return self.cache.backend.asr_path(path)

    @classmethod
    def find_root(cls, path=Path()) -> 'Repository':
        path = Path(path).absolute()
        origpath = path
        while not (path / ASR_DIR).is_dir():
            if path == Path('/'):
                raise ASRRootNotFound(error_not_initialized
                                      .format(directory=origpath))
            path = path.parent
        assert (path / ASR_DIR).is_dir()
        return cls(path)

    @classmethod
    def root_is_initialized(cls) -> bool:
        try:
            cls.find_root()
        except ASRRootNotFound:
            return False
        return True

    @classmethod
    def initialize(cls, root: Path) -> 'Repository':
        asr_dir = root / ASR_DIR

        if asr_dir.exists():
            raise FileExistsError(f'Repository already initialized at {root}')

        asr_dir.mkdir()
        return cls(root)

    def __repr__(self):
        return f'{type(self).__name__}(root={self.root})'


def find_root(path: str = '.'):
    return Repository.find_root(path).root


ASR_DIR = Path('.asr')
