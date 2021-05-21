import pathlib


class ASRRootNotFound(Exception):
    pass


def root_is_initialized():
    try:
        find_root()
        return True
    except ASRRootNotFound:
        return False


def initialize_root(directory: pathlib.Path = pathlib.Path('.')):
    asr_dir = directory / ASR_DIR
    assert not asr_dir.exists()
    asr_dir.mkdir()


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
