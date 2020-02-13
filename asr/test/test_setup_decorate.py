import pytest
from .conftest import test_materials

from ase.build import bulk

Si = bulk('Si')


@pytest.mark.ci
@pytest.mark.parametrize("atoms", [Si] + test_materials)
def test_setup_decorate(separate_folder, usemocks, atoms):
    from asr.setup.decorate import main
    from ase.io import write
    from ase.db import connect
    from pathlib import Path
    write('structure.json', atoms)
    main(atoms='structure.json')

    assert Path('decorated.db').is_file()

    db = connect('decorated.db')
    if all(map(lambda x: x == 'Si', atoms.get_chemical_symbols())):
        db.get('Ge')
