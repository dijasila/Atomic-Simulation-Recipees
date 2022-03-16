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
import warnings

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


def dct_to_object(dct):
    """Convert dictionary to object."""
    from .results import decode_object, UnknownDataFormat
    warnings.warn(
        "asr.core.dct_to_object is renamed to asr.core.decode_object. "
        "This function will be removed in a future release."
        "Please update your scripts accordingly.",
        DeprecationWarning)

    try:
        obj = decode_object(dct)
        return obj
    except UnknownDataFormat:
        assert isinstance(dct, dict), 'Cannot convert dct to object!'
        return dct


def read_file(filename: str) -> str:
    return Path(filename).read_text()


def decode_json(text: str) -> dict:
    dct = jsonio.decode(text)
    return dct


def read_json(filename):
    """Read json file."""
    from .results import decode_object
    text = read_file(filename)
    literal_object = decode_json(text)
    obj = decode_object(literal_object)
    return obj


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


def get_recipe_from_name(name):
    # Get a recipe from a name like asr.gs@postprocessing
    import importlib
    assert name.startswith('asr.'), \
        'Not allowed to load recipe from outside of ASR.'
    mod, func = parse_mod_func(name)
    module = importlib.import_module(mod)
    return getattr(module, func)


def parse_mod_func(name):
    # Split a module function reference like
    # asr.relax@main into asr.relax and main.
    mod, *func = name.split('@')
    if not func:
        func = ['main']

    assert len(func) == 1, \
        'You cannot have multiple : in your function description'

    return mod, func[0]


def get_dep_tree(name, reload=True):
    # Get the tree of dependencies from recipe of "name"
    # by following dependencies of dependencies
    import importlib

    tmpdeplist = [name]

    for i in range(1000):
        if i == len(tmpdeplist):
            break
        dep = tmpdeplist[i]
        mod, func = parse_mod_func(dep)
        module = importlib.import_module(mod)

        assert hasattr(module, func), f'{module}.{func} doesn\'t exist'
        function = getattr(module, func)
        dependencies = function.dependencies
        # if not dependencies and hasattr(module, 'dependencies'):
        #     dependencies = module.dependencies

        for dependency in dependencies:
            tmpdeplist.append(dependency)
    else:
        raise AssertionError('Unreasonably many dependencies')

    tmpdeplist.reverse()
    deplist = []
    for dep in tmpdeplist:
        if dep not in deplist:
            deplist.append(dep)

    return deplist
