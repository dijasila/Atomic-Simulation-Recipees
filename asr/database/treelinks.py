from asr.core import (command, option, ASRResult, prepare_result,
                      read_json, CommaStr)
from pathlib import Path
import typing


@prepare_result
class Result(ASRResult):
    """Container for treelinks result."""

    links: typing.List[str]

    key_descriptions: typing.Dict[str, str] = dict(
        links='List of uids to link to.'
    )


@command('asr.database.treelinks')
@option('--include', help='Comma-separated string of folders to include.',
        type=CommaStr())
@option('--exclude', help='Comma-separated string of folders to exclude.',
        type=CommaStr())
def main(include: str = '',
         exclude: str = '') -> Result:
    """Create links.json based on the tree-structure.

    Choose the respective option to choose, which kind of tree is
    currently present. !Change that description!
    """
    p = Path('.')

    folders = recursive_through_folders(p, include, exclude)
    links = create_tree_links(folders)

    return Result.fromdata(links=links)


def write_links(path, link_uids):
    from asr.core import write_json
    write_json(path / 'links.json', link_uids)


def recursive_through_folders(path, include, exclude):
    """Go through folders recursively to find folders.

    Find folders with 'include' specifically and exclude folders with 'exclude'.
    """
    import os

    atomfile = 'structure.json'
    folders = []

    for root, dirs, files in os.walk(path, topdown=True, followlinks=False):
        for add in include:
            for dir in dirs:
                if dir == add:
                    folder = Path(root + '/' + dir)
                    if Path(folder / atomfile).is_file():
                        folders.append(folder)

    return folders


def create_tree_links(folders):
    """Return a list of structure uids to link to.

    Based on the tree structure created from setup.defects.
    """
    uids = []
    for folder in folders:
        fingerprint_res = read_json(folder
                                    / 'results-asr.database.material_fingerprint.json')
        uid = fingerprint_res['uid']
        uids.append(uid)

    fingerprint_res = read_json(folder
                                / 'results-asr.database.material_fingerprint.json')
    uid = fingerprint_res['uid']
    uids.append(uid)

    links = {'uids': uids}

    write_links(folder, links)

    return uids
