import pathlib


class ASRRootNotFound(Exception):
    pass


def root_is_initialized():
    try:
        find_root()
        return True
    except ASRRootNotFound:
        return False


def initialize_root():
    assert not ASR_DIR.exists()
    ASR_DIR.mkdir()


def find_root(path: str = '.'):
    path = pathlib.Path(path).absolute()
    while not (path / ASR_DIR).is_dir():
        if path == pathlib.Path('/'):
            raise ASRRootNotFound
        path = path.parent
    assert (path / ASR_DIR).is_dir()
    return path


def find_repo_root(path: str = '.'):
    root = find_root()
    return root / ASR_DIR


ASR_DIR = pathlib.Path('.asr')
