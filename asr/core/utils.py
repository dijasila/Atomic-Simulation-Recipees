"""Implements useful utility functions needed for several asr features.

Functions
---------

    parse_dict_string: Convert a string-serialized dict, return a real dict.

"""
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Union, List

import numpy as np
from ase.io import jsonio
import ase.parallel as parallel
from ast import literal_eval


def parse_dict_string(string, dct=None):
    """Convert a string-serialized dict, return a real dict."""
    if dct is None:
        dct = {}

    # Locate ellipsis
    string = string.replace('...', 'None:None')
    tmpdct = literal_eval(string)
    recursive_update(tmpdct, dct)
    return tmpdct


def recursive_update(dct, defaultdct):
    """Recursively update defualtdct with values from dct."""
    if None in dct:
        # This marks that we take default values from defaultdct
        del dct[None]
        for key in defaultdct:
            if key not in dct:
                dct[key] = defaultdct[key]

    for key, value in dct.items():
        if isinstance(value, dict) and None in value:
            if key not in defaultdct:
                del value[None]
                continue
            if not isinstance(defaultdct[key], dict):
                del value[None]
                continue
            recursive_update(dct[key], defaultdct[key])


def md5sum(filename):
    from hashlib import md5
    hash = md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * hash.block_size), b""):
            hash.update(chunk)
    return hash.hexdigest()


@contextmanager
def chdir(folder, create=False, empty=False):
    dir = os.getcwd()
    if empty and folder.is_dir():
        import shutil
        shutil.rmtree(str(folder))
    if create and not folder.is_dir():
        os.mkdir(folder)
    os.chdir(str(folder))
    yield
    os.chdir(dir)


def encode_json(data):
    from ase.io.jsonio import MyEncoder
    return MyEncoder(indent=1).encode(data)


def write_json(filename, data):
    write_file(filename, encode_json(data))


def write_file(filename, text):
    from pathlib import Path
    from ase.parallel import world

    with file_barrier([filename]):
        if world.rank == 0:
            Path(filename).write_text(text)


def read_json(filename):
    from pathlib import Path
    dct = jsonio.decode(Path(filename).read_text())
    return dct


def unlink(path: Union[str, Path], world=None):
    """Safely unlink path (delete file or symbolic link)."""
    if isinstance(path, str):
        path = Path(path)
    if world is None:
        world = parallel.world

    world.barrier()
    # Remove file:
    if world.rank == 0:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    else:
        while path.is_file():
            time.sleep(1.0)
    world.barrier()


@contextmanager
def file_barrier(paths: List[Union[str, Path]], world=None,
                 delete=True):
    """Context manager for writing a file.

    After the with-block all cores will be able to read the file.

    Do "with file_barrier(['something.txt']):"

    This will remove the file, write the file and wait for the file.
    """
    if world is None:
        world = parallel.world

    for i, path in enumerate(paths):
        if isinstance(path, str):
            path = Path(path)
            paths[i] = path
        # Remove file:
        if delete:
            unlink(path, world)

    yield

    # Wait for file:
    i = 0
    world.barrier()
    while not all([path.is_file() for path in paths]):
        filenames = ', '.join([path.name for path in paths
                               if not path.is_file()])
        if i > 0:
            print(f'Waiting for ~{i}sec on existence of {filenames}'
                  ' on all ranks')
        time.sleep(1.0)
        i += 1
    world.barrier()


def singleprec_dict(dct):
    assert isinstance(dct, dict), f'Input {dct} is not dict.'

    for key, value in dct.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.int64:
                value = value.astype(np.int32)
            elif value.dtype == np.float64:
                value = value.astype(np.float32)
            elif value.dtype == np.complex128:
                value = value.astype(np.complex64)
            dct[key] = value
        elif isinstance(value, dict):
            dct[key] = singleprec_dict(value)

    return dct
