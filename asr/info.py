"""Manually set key-value-pairs for material."""

from asr.core import command, argument, read_json, write_json
import click
from pathlib import Path
from typing import List, Tuple


class KeyValuePair(click.ParamType):
    """Read atoms object from filename and return Atoms object."""

    def convert(self, value, param, ctx):
        """Convert string to a (key, value) tuple."""
        assert ':' in value
        key, value = value.split(':')
        return key, value


protected_keys = {'material_type': {'primary', 'secondary'}}


def check_key_value(key, value):
    """Check validity of any protected key value pairs."""
    if key in protected_keys:
        allowed_values = protected_keys[key]
        if value not in allowed_values:
            raise ValueError(
                f'Protected {key}={value} not in allowed values: '
                f'{allowed_values}.'
            )


@command('asr.info')
@argument('key_value_pairs', metavar='key:value', nargs=-1,
          type=KeyValuePair())
def main(key_value_pairs: List[Tuple[str, str]]):
    """Set additional key value pairs.

    Some key valye pairs are protected and can assume a limited set of
    values::

        - `material_type`: `primary`, `secondary`.

    These extra key value pairs are stored in info.json.

    """
    infofile = Path('info.json')
    if infofile.is_file():
        info = read_json(infofile)
    else:
        info = {}

    for key, value in key_value_pairs:
        check_key_value(key, value)
        if value == '':
            info.pop(key, None)
        else:
            info[key] = value
    write_json(infofile, info)
