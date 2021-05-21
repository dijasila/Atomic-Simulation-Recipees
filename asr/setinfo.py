"""Manually set key-value-pairs for material."""

from asr.core import read_json, write_json
from pathlib import Path
from typing import List, Tuple


protected_keys = {'first_class_material': {True, False}}


def check_key_value(key, value):
    """Check validity of any protected key value pairs."""
    if key in protected_keys:
        allowed_values = protected_keys[key]
        if value not in allowed_values:
            raise ValueError(
                f'Protected {key}={value} not in allowed values: '
                f'{allowed_values}.'
            )


def main(key_value_pairs: List[Tuple[str, str]]):
    """Set additional key value pairs.

    These extra key value pairs are stored in info.json.  To set a key
    value pair simply do::

        asr run "setinfo key1:'mystr' key2:1 key3:True"

    The values supplied values will be interpred and the result will
    be {'key1': 'mystr', 'key2': 1, 'key3': True}

    Some key value pairs are protected and can assume a limited set of
    values::

        - `first_class_material`: True, False

    To delete an existing key-value-pair in info.json supply an empty
    string as a value, i.e.:

    asr run "setinfo mykey:"

    would delete "mykey".

    """
    infofile = Path('info.json')
    if infofile.is_file():
        info = read_json(infofile)
    else:
        info = {}

    for key, value in key_value_pairs:
        if value == '':
            info.pop(key, None)
        else:
            check_key_value(key, value)
            info[key] = value

    write_json(infofile, info)
