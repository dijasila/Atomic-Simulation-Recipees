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


class CommaStr(click.ParamType):
    """Read in a comma-separated strings and return a list of strings."""

    name = "comma_string"

    def convert(self, value, param, ctx):
        """Convert string with commas to list of strings."""
        if isinstance(value, str):
            return value.split(',')


class DictStr(click.ParamType):
    """Read atoms object from filename and return Atoms object."""

    name = "dictionary_string"

    def convert(self, value, param, ctx):
        """Convert string to a dictionary."""
        if isinstance(value, dict):
            return value
        return parse_dict_string(value)


def clickify_docstring(doc):
    """Take a standard docstring a make it Click compatible."""
    if doc is None:
        return
    doc_n = doc.split('\n')
    clickdoc = []
    skip = False
    for i, line in enumerate(doc_n):
        if skip:
            skip = False
            continue
        lspaces = len(line) - len(line.lstrip(' '))
        spaces = ' ' * lspaces
        bb = spaces + '\b'
        if line.endswith('::'):
            skip = True

            if not doc_n[i - 1].strip(' '):
                clickdoc.pop(-1)
                clickdoc.extend([bb, line, bb])
            else:
                clickdoc.extend([line, bb])
        elif ('-' in line
              and (spaces + '-' * (len(line) - lspaces)) == line):
            clickdoc.insert(-1, bb)
            clickdoc.append(line)
        else:
            clickdoc.append(line)
    doc = '\n'.join(clickdoc)

    return doc
