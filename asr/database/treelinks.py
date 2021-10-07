import os
import typing
from pathlib import Path
from fnmatch import fnmatch
from asr.core import ASRResult, prepare_result
from ase.io import read
from asr.database.material_fingerprint import main as material_fingerprint


ATOMFILE = "structure.json"


@prepare_result
class Result(ASRResult):
    """Container for treelinks result."""

    links: typing.List[str]

    key_descriptions: typing.Dict[str, str] = dict(links="List of uids to link to.")


def main(include: str = "", exclude: str = "") -> Result:
    """Create links.json based on the tree-structure.

    Recipe to create links.json files based on the tree structure of
    your current directory. Goes through the directories recursively
    and adds repective uids if a 'structure.json' is present. Give
    '--include' option for folders to specifically include, and
    '--exclude' option for folders to exclude. Note, that you can't
    use both '--include' and '--exclude' at the same time!
    """
    p = Path(".")

    folders = recursive_through_folders(p, include, exclude)
    if folders != []:
        links = create_tree_links(folders)
        print("INFO: Created links based on the tree structure.")
    else:
        links = []
        print("INFO: No links created based on the tree structure.")
    print("INFO: Added links to links.json.")

    return Result.fromdata(links=links)


def write_links(path, link_uids):
    from asr.core import read_json, write_json

    newlinks = []
    linkspath = Path(path / "links.json")
    if linkspath.is_file():
        oldlinks = read_json(linkspath)["uids"]
    else:
        oldlinks = []

    for link_uid in link_uids["uids"]:
        if link_uid not in oldlinks:
            newlinks.append(link_uid)

    if oldlinks != []:
        for link in oldlinks:
            newlinks.append(link)
    link_uids = {"uids": newlinks}
    write_json(path / "links.json", link_uids)

    return None


def recursive_through_folders(path, include, exclude):
    """Go through folders recursively to find folders.

    Find folders with 'include' specifically and exclude folders with 'exclude'.
    """
    folders = []

    if exclude == [""]:
        for root, dirs, files in os.walk(path, topdown=True, followlinks=False):
            for dir in dirs:
                for add in include:
                    rootlist = root.split("/")
                    if (
                        fnmatch(dir, add)
                        or include == [""]
                        or any(fnmatch(rootstring, add) for rootstring in rootlist)
                    ):
                        folder = Path(root + "/" + dir)
                        if Path(folder / ATOMFILE).is_file():
                            folders.append(folder)
    elif include == [""] and exclude != [""]:
        for root, dirs, files in os.walk(path, topdown=True, followlinks=False):
            for dir in dirs:
                for rm in exclude:
                    rootlist = root.split("/")
                    if not fnmatch(dir, rm) and not any(
                        fnmatch(rootstring, rm) for rootstring in rootlist
                    ):
                        folder = Path(root + "/" + dir)
                        if Path(folder / ATOMFILE).is_file():
                            folders.append(folder)
    elif include != "" and exclude != "":
        raise AssertionError(
            "It is not possible to give both an include and exclude "
            "list as input to asr.database.treelinks!"
        )

    return folders


def create_tree_links(folders):
    """Return a list of structure uids to link to.

    Creates uids for structures within the given folder list.
    """
    print("INFO: Create links for the following folders:")

    parent_uid = material_fingerprint(atoms=read(ATOMFILE))["uid"]
    uids = [parent_uid]

    for folder in folders:
        fingerprint_res = material_fingerprint(atoms=read(folder / ATOMFILE))
        uid = fingerprint_res["uid"]
        if uid not in uids:
            uids.append(uid)
            print(f"      {folder.absolute()} -> uid: {uid}")

    links = {"uids": uids}

    for folder in folders:
        write_links(folder, links)

    return uids
