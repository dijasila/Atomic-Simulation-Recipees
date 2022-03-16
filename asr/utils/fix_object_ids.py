from pathlib import Path
from asr.core import read_file, decode_json, decode_result, write_file
from asr.core.results import ModuleNameIsCorrupt, UnknownDataFormat
import copy
from typing import List
import click
import re


def extract_recipe_from_filename(filename: str):
    """Parse filename and return recipe name."""
    pattern = re.compile(r'results-(.*)\.json')  # noqa
    m = pattern.match(filename)
    return m.group(1)


def fix_object_id(filename: str, dct: dict):

    assert filename.startswith('results-asr.')
    assert filename.endswith('.json')
    for key, value in dct.items():
        if isinstance(value, dict):
            value = fix_object_id(filename, value)
            dct[key] = value

    if 'object_id' not in dct:
        return dct

    recipename = extract_recipe_from_filename(filename)
    if '@' in recipename:
        recipemodule = recipename.split('@')[0]
    else:
        recipemodule = recipename
    object_id = dct['object_id']
    dct = copy.copy(dct)
    modulename, objectname = object_id.split('::')
    new_object_id = f'{recipemodule}::{objectname}'
    dct['object_id'] = new_object_id
    dct['constructor'] = new_object_id
    return dct


def _fix_folders(folders):
    for folder in folders:
        folder = Path(folder).absolute()
        print(f'Checking folder={folder}')
        for path in folder.glob('results-*.json'):
            text = read_file(path)
            dct = decode_json(text)
            filename = path.name
            try:
                decode_result(dct)
            except ModuleNameIsCorrupt:
                dct = fix_object_id(filename, dct)
                result = decode_result(dct)
                print(f'Fixing bad file: {filename}')
                json_string = result.format_as('json')
                write_file(path, json_string)
            except UnknownDataFormat:
                # Assume that we have to insert __asr_name__
                # since this is a _very_ old results file.
                recipename = extract_recipe_from_filename(filename)
                dct['__asr_name__'] = recipename
                try:
                    result = decode_result(dct)
                    print(f'Fixing missing __asr_name__ in file: {filename}')
                    json_string = result.format_as('json')
                    write_file(path, json_string)
                except Exception:
                    print(
                        'Located problematic results file that '
                        f'could not be fixed: {path}')


@click.command()
@click.argument('folders', nargs=-1)
def fix_folders(folders: List[str]):
    """Fix folders constaining bad result files.

    Substitutes __main__ in object ids with actual module names.
    """
    _fix_folders(folders)


if __name__ == '__main__':
    fix_folders()
