"""Implement custom types for ASR."""
import click
from ase.io import read
from ase.io.formats import UnknownFileTypeError
from asr.core import parse_dict_string


class AtomsFile(click.ParamType):
    """Read atoms object from filename and return Atoms object."""

    name = "atomsfile"

    def __init__(self, must_exist=True, *args, **kwargs):
        """Initialize AtomsFile object.

        Parameters
        ----------
        must_exist : bool
            If False, errors relating to empty or missing files will be
            ignored and the returned atoms will be None in that case. If True,
            all errors will be raised if encountered.

        Returns
        -------
        atoms : `ase.Atoms`
        """
        self.must_exist = must_exist
        super().__init__(*args, **kwargs)

    def convert(self, value, param, ctx):
        """Convert string to atoms object."""
        try:
            return read(value, parallel=False)
        except (IOError, UnknownFileTypeError, StopIteration):
            if self.must_exist:
                raise
            return None


class DictStr(click.ParamType):
    """Read atoms object from filename and return Atoms object."""

    name = "dictionary_string"

    def convert(self, value, param, ctx):
        """Convert string to a dictionary."""
        if isinstance(value, dict):
            return value
        return parse_dict_string(value)
