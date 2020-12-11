import os
import typing
from pathlib import Path
from fnmatch import fnmatch
from asr.core import (command, option, ASRResult, prepare_result,
                      read_json, CommaStr)


# TODO: implement write_links() function properly


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
    if folders != []:
        links = create_tree_links(folders)
        print('INFO: Created links based on the tree structure.')
    else:
        links = []
        print('INFO: No links created based on the tree structure.')
    print('INFO: Added links to links.json.')

    return Result.fromdata(links=links)


def write_links(path, link_uids):
    """NEEDS TO BE PROPERLY IMPLEMENTED!"""
    from asr.core import write_json  # , read_json
    # if Path(path / 'links.json').is_file():
    #     links = read_json(Path(path / 'links.json'))['uids']
    #     assert isinstance(links, list), 'links.json broken!'
    #     print('INFO: Old links: {}'.format(links))
    #     for link in link_uids['uids']:
    #         if link not in links:
    #             links.append(link)
    #     print('INFO: Updated links: {}'.format(links))
    # else:
    #     links = link_uids

    write_json(path / 'links.json', link_uids)


def recursive_through_folders(path, include, exclude):
    """Go through folders recursively to find folders.

    Find folders with 'include' specifically and exclude folders with 'exclude'.
    """
    atomfile = 'structure.json'
    folders = []

    if exclude == ['']:
        for root, dirs, files in os.walk(path, topdown=True, followlinks=False):
            for dir in dirs:
                for add in include:
                    rootlist = root.split('/')
                    if (fnmatch(dir, add) or include == ['']
                       or any(fnmatch(rootstring, add)
                              for rootstring in rootlist)):
                        folder = Path(root + '/' + dir)
                        if Path(folder / atomfile).is_file():
                            folders.append(folder)
    elif include == [''] and exclude != ['']:
        for root, dirs, files in os.walk(path, topdown=True, followlinks=False):
            for dir in dirs:
                for rm in exclude:
                    rootlist = root.split('/')
                    if (not fnmatch(dir, rm)
                       and not any(fnmatch(rootstring, rm)
                                   for rootstring in rootlist)):
                        folder = Path(root + '/' + dir)
                        if Path(folder / atomfile).is_file():
                            folders.append(folder)
    elif include != [''] and exclude != ['']:
        raise AssertionError('It is not possible to give both an include and exclude '
                             'list as input to asr.database.treelinks!')

    return folders


def create_tree_links(folders):
    """Return a list of structure uids to link to.

    Based on the tree structure created from setup.defects.
    """
    print('INFO: Create links for the following folders:')

    uids = []
    for folder in folders:
        fingerprint_res = read_json(folder
                                    / 'results-asr.database.material_fingerprint.json')
        uid = fingerprint_res['uid']
        if uid not in uids:
            uids.append(uid)
            print(f"      {folder.absolute()} -> uid: {uid}")

    links = {'uids': uids}

    write_links(folder, links)

    return uids
