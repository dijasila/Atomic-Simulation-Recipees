"""Manually set key-value-pairs for material."""

from asr.core import command, argument, read_json, write_json
import click
from pathlib import Path
from typing import List


class KeyValuePair(click.ParamType):
    """Read atoms object from filename and return Atoms object."""

    def convert(self, value, param, ctx):
        """Convert string to a (key, value) tuple."""
        assert ':' in value
        key, value = value.split(':')
        return key, value


@command('asr.info')
@argument('key_value_pairs', metavar='key:value', nargs=-1,
          type=KeyValuePair())
def main(key_value_pairs: List):
    """Set additional key value pairs."""
    infofile = Path('info.json')
    if infofile.is_file():
        info = read_json(infofile)
    else:
        info = {}

    for key, value in key_value_pairs:
        if value == '':
            info.pop(key, None)
        else:
            info[key] = value
    write_json(infofile, info)
