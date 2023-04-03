import os
from pathlib import Path
import pytest
from .materials import std_test_materials, BN
from ase.db import connect
from asr.database.totree import main


@pytest.mark.xfail
@pytest.mark.ci
def test_database_totree(unlocked_repo):
    dbname = 'database.db'
    db = connect(dbname)
    for atoms in std_test_materials:
        db.write(atoms)

    tree = unlocked_repo.root / 'tree'
    assert tree.is_dir()

    main(database=dbname,
         tree_structure='tree/{stoi}/{spg}/{formula:abc}')

    assert len(list(tree.glob('*'))) == 0  # No files after dry run

    main(database=dbname, run=True,
         tree_structure='{stoi}/{spg}/{formula:abc}')

    for nesting in ['A/123/Ag',
                    'A/227/Si2',
                    'AB/187/BN']:

        # XXX The define-string is liable to change
        structuredirs = list((tree / nesting).glob('define-*'))
        assert len(structuredirs) == 1
        structurefile = structuredirs[0] / 'output.json'
        assert structurefile.is_file()


@pytest.fixture
def make_test_db(asr_tmpdir):
    """Make a database that contains data in various forms."""
    dbname = 'database.db'
    db = connect(dbname)
    p = Path('hardlinkedfile.txt')
    p.write_text('Some content.')

    data = {'file.json': {'key': 'value'},
            'hardlinkedfile.txt': {'pointer': str(p.absolute())}}

    db.write(BN, data=data)
    return db


@pytest.mark.xfail
@pytest.mark.ci
def test_database_totree_files_and_hard_links(make_test_db):
    """Test that hard links are correctly reproduced."""
    from asr.core import read_json

    dbname = 'database.db'
    main(database=dbname, run=True, copy=True, atomsfile='structure.json',
         tree_structure='tree/{stoi}/{spg}/{formula:abc}')
    hardlink = Path('tree/AB/187/BN/hardlinkedfile.txt')
    filejson = Path('tree/AB/187/BN/file.json')
    assert Path('tree/AB/187/BN/structure.json').is_file()
    assert filejson.is_file()
    assert hardlink.is_file()

    contents = read_json(filejson)
    assert contents['key'] == 'value'

    # Check that link is not symlink
    assert not os.path.islink(hardlink)
    assert os.stat(hardlink).st_ino == os.stat('hardlinkedfile.txt').st_ino
