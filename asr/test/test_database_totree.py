import pytest
from .conftest import test_materials


@pytest.mark.ci
def test_database_totree(separate_folder):
    from ase.db import connect
    from asr.database.totree import main
    from pathlib import Path

    dbname = 'database.db'
    db = connect(dbname)
    for atoms in test_materials:
        db.write(atoms)

    main(database=dbname)

    assert not Path('tree').is_dir()

    main(database=dbname, run=True)

    assert Path('tree').is_dir()
