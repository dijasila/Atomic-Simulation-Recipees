from asr.core import (command, option, ASRResult, prepare_result,
                      read_json, CommaStr)
from pathlib import Path
import typing


# TODO: write general write_links function and put into core functionality
# TODO: implement more create_links functions for other projects


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
    currently present.
    """
    p = Path('.')

    links = create_tree_links(path=p, include, exclude)

    return Result.fromdata(links=links)


def write_links(path, link_uids):
    from asr.core import write_json
    write_json(path / 'links.json', link_uids)


def create_defect_links(path):
    """Return a list of structure uids to link to.

    Based on the tree structure created from setup.defects.
    """
    parent = Path(path.absolute() / '../../../')

    folders = list(parent.glob('**/charge_0/'))
    uids = []
    for folder in folders:
        fingerprint_res = read_json(folder
                                    / 'results-asr.database.material_fingerprint.json')
        uid = fingerprint_res['uid']
        uids.append(uid)

    fingerprint_res = read_json(parent
                                / 'results-asr.database.material_fingerprint.json')
    uid = fingerprint_res['uid']
    uids.append(uid)

    links = {'uids': uids}

    write_links(path, links)

    return uids
