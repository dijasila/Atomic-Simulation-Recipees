"""Tests of asr.database.from_tree."""

from asr.core import chdir
from ase.io import read
import pytest
import os
from pathlib import Path
from .materials import Si, Fe


@pytest.fixture
def folder_tree():
    """Set up a tree like folder structure."""
    from asr.setup.displacements import main as displacements
    folders = [('materials/Si2', Si),
               ('materials/Si2/repeated', Si.repeat((2, 1, 1))),
               ('materials/Fe', Fe)]

    for folder, atoms in folders:
        os.makedirs(folder)
        atoms.write(Path(folder) / 'structure.json')

    with chdir('materials/Si2/'):
        displacement_results = displacements()
        displacement_folders = [
            ('materials/Si2/' + folder, read(Path(folder) / 'structure.json'))
            for folder in displacement_results['folders']
        ]

    folders.extend(displacement_folders)
    return folders


def make_tree(folder: str):
    tree = set()
    for root, dirs, files in os.walk(folder, topdown=True, followlinks=False):
        for filename in [''] + files:
            name = Path(root) / filename
            rel_path = str(name.relative_to(folder))
            tree.add(rel_path)
    return tree


@pytest.mark.ci
def test_database_fromtree_totree(asr_tmpdir, folder_tree):
    """Make sure a database can be packed and unpacked faithfully."""
    from asr.database.fromtree import main as fromtree
    from asr.database.totree import main as totree
    from ase.db import connect

    folders = [folder[0] for folder in folder_tree]
    fromtree(folders=['materials'], recursive=True)

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


@pytest.mark.ci
def test_database_fromtree_raises_when_missing_uids(asr_tmpdir, folder_tree):
    """Make sure a database can be packed and unpacked faithfully."""
    from asr.database.fromtree import main as fromtree
    from asr.database.fromtree import MissingUIDS
    with pytest.raises(MissingUIDS):
        fromtree(folders=['materials/Si2'])
