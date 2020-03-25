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
    assert Path('tree/A/123/Ag-A-123-c-0/structure.json').is_file()
    assert Path('tree/A/227/Si2-A-227-b-0/structure.json').is_file()
    assert Path('tree/AB/187/BN-AB-187-a-c-0/structure.json').is_file()
