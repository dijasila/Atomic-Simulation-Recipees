"""Tests of asr.database.from_tree."""

from asr.core import chdir
import pytest
import os
from pathlib import Path
from .materials import Si, Fe


@pytest.fixture
def folder_tree(asr_tmpdir):
    """Set up a tree like folder structure."""
    from asr.setup.displacements import main as displacements
    folders = [('materials/Si2', Si),
               ('materials/Si2/repeated', Si.repeat((2, 1, 1))),
               ('materials/Fe', Fe)]

    for folder, atoms in folders:
        os.makedirs(folder)
        atoms.write(Path(folder) / 'structure.json')

    with chdir(Path('materials/Si2/displaced/'), create=True):
        displacement_results = displacements(
            atoms=Si, displacement=0.01)
        displaced_atoms = displacement_results[0][-1]
        displaced_atoms.write('structure.json')
        displacement_folders = [('materials/Si2/displaced', displaced_atoms)]

    folders.extend(displacement_folders)
    return folders


def make_tree(folder: str):
    tree = set()
    for root, dirs, files in os.walk(folder, topdown=True, followlinks=False):
        if '.asr' in root:
            continue
        for filename in [''] + files:
            name = Path(root) / filename
            rel_path = str(name.relative_to(folder))
            tree.add(rel_path)
    return tree


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
@pytest.mark.parametrize('njobs', [1, 2, 8])
def test_database_fromtree_totree(folder_tree, njobs):
    """Make sure a database can be packed and unpacked faithfully."""
    from asr.database.fromtree import main as fromtree
    from asr.database.totree import main as totree
    from asr.database import connect

    folders = [folder[0] for folder in folder_tree]
    fromtree(folders=['materials'], recursive=True, njobs=njobs)

    db = connect('database.db')
    assert len(db) == len(folders)

    row = db.get(folder=folder_tree[0][0])
    childfolders = []
    for folder in folder_tree[1:]:
        try:
            child = str(Path(folder[0]).relative_to('materials/Si2'))
            childfolders.append(child)
        except ValueError:
            pass

    children = row.data['__children__']
    assert set(childfolders) == set(children)
    totree('database.db', tree_structure='tree/{row.formula}',
           run=True)
    tree1 = make_tree('materials')
    tree2 = make_tree('tree')
    assert {'.', 'Si2'}.issubset(tree1)
    assert tree1 == tree2


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
def test_database_fromtree_raises_when_missing_uids(asr_tmpdir, folder_tree):
    """Make sure a database can be packed and unpacked faithfully."""
    from asr.database.fromtree import main as fromtree
    from asr.database.fromtree import MissingUIDS
    with pytest.raises(MissingUIDS):
        fromtree(folders=['materials/Si2'])


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
def test_database_fromtree_works_without_write_permission(
        asr_tmpdir, folder_tree):
    from asr.database.fromtree import main as fromtree
    import stat
    os.chmod(folder_tree[0][0], stat.S_IRUSR | stat.S_IXUSR)
    try:
        fromtree(folders=['materials/Si2'], recursive=True)
    finally:
        os.chmod(folder_tree[0][0], stat.S_IRUSR | stat.S_IXUSR | stat.S_IWUSR)
