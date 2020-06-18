"""Tests of asr.database.from_tree."""

from asr.core import chdir
from ase.io import read
import pytest
import os
from pathlib import Path
from .materials import Si


@pytest.fixture
def folder_tree():
    """Set up a tree like folder structure."""
    from asr.setup.displacements import main as displacements
    folders = [('Si/', Si),
               ('Si/repeated', Si.repeat((2, 1, 1)))]

    for folder, atoms in folders:
        os.makedirs(folder)
        atoms.write(Path(folder) / 'structure.json')

    with chdir('Si/'):
        displacement_results = displacements()
        displacement_folders = [
            ('Si/' + folder, read(Path(folder) / 'structure.json'))
            for folder in displacement_results['folders']
        ]

    folders.extend(displacement_folders)
    return folders


@pytest.mark.ci
def test_database_fromtree(asr_tmpdir, folder_tree):
    from asr.database.fromtree import main as fromtree
    from ase.db import connect

    folders = [folder_tree[0][0]]
    fromtree(folders=folders)

    db = connect('database.db')
    assert len(db) == 1

    row = db.get(1)
    linkfolders = [folder[0][3:] for folder in folder_tree[1:]]
    links = row.data['__links__']
    assert set(linkfolders) == set(links)
