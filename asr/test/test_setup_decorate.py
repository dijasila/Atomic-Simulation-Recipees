import pytest
from ase.build import bulk

Si = bulk('Si')


@pytest.mark.ci
def test_setup_decorate_si(separate_folder, mockgpaw):
    from asr.setup.decorate import main
    from ase.io import write
    from ase.db import connect
    from pathlib import Path
    write('structure.json', Si)
    main(atoms='structure.json')
    assert Path('decorated.db').is_file()
    db = connect('decorated.db')
    db.get('Ge')
