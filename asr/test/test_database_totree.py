import pytest
from .materials import std_test_materials


@pytest.mark.ci
def test_database_totree(asr_tmpdir_w_params):
    from ase.db import connect
    from asr.database.totree import main
    from pathlib import Path

    dbname = 'database.db'
    db = connect(dbname)
    for atoms in std_test_materials:
        db.write(atoms)

    main(database=dbname)

    assert not Path('tree').is_dir()

    main(database=dbname, run=True)

    assert Path('tree').is_dir()
    assert Path('tree/A/123/Ag-A-123-c-0/structure.json').is_file()
    assert Path('tree/A/227/Si2-A-227-b-0/structure.json').is_file()
    assert Path('tree/AB/187/BN-AB-187-b-d-0/structure.json').is_file()
