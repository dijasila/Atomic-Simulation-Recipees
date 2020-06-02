"""Implement custom types for ASR."""
import click
from ase.io import read


class AtomsFileParamType(click.ParamType):
    """Read atoms object from filename and return Atoms object."""

    name = "atomsfile"

    def convert(self, value, param, ctx):
        """Convert string to atoms object."""
        return read(value)


AtomsFile = AtomsFileParamType()
