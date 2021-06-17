import pytest
from ase.db import connect
# from .materials import std_test_materials
from pathlib import Path

# @pytest.fixture(params=std_test_materials)
# def duplicates_test_db(request, asr_tmpdir):
#     """Set up a database containing only duplicates of BN."""
#     import numpy as np
#     import ase.db

#     db = ase.db.connect("duplicates.db")
#     atoms = request.param.copy()

#     db.write(atoms=atoms)

#     rotated_atoms = atoms.copy()
#     rotated_atoms.rotate(23, v='z', rotate_cell=True)
#     db.write(atoms=rotated_atoms, magstate='FM')

#     pbc_c = atoms.get_pbc()
#     repeat = np.array([2, 2, 2], int)
#     repeat[~pbc_c] = 1
#     supercell_ref = atoms.repeat(repeat)
#     db.write(supercell_ref)

#     translated_atoms = atoms.copy()
#     translated_atoms.translate(0.5)
#     db.write(translated_atoms)

#     rattled_atoms = atoms.copy()
#     rattled_atoms.rattle(0.001)
#     db.write(rattled_atoms)

#     stretch_nonpbc_atoms = atoms.copy()
#     cell = stretch_nonpbc_atoms.get_cell()
#     pbc_c = atoms.get_pbc()
#     cell[~pbc_c][:, ~pbc_c] *= 2
#     stretch_nonpbc_atoms.set_cell(cell)
#     db.write(stretch_nonpbc_atoms)

#     return (atoms, db)


@pytest.mark.ci
def test_database_duplicates(duplicates_test_db):
    from asr.database.duplicates import main

    results = main(
        connect('duplicates.db'),
        connect('duplicates_removed.db'),
        filterstring='<=natoms,<id')

    assert Path('duplicates_removed.db').is_file()
    nduplicates = len(duplicates_test_db[1])
    duplicate_groups = results['duplicate_groups']
    assert duplicate_groups[0]['include'] == [1]
    assert duplicate_groups[0]['exclude'] == list(range(2, nduplicates + 1))


@pytest.mark.ci
def test_database_duplicates_filter_magstate(duplicates_test_db):
    from asr.database.duplicates import main

    results = main(
        connect('duplicates.db'),
        connect('duplicates_removed.db'),
        comparison_keys='magstate',
        filterstring='<=natoms,<id',
    )

    duplicate_groups = results['duplicate_groups']
    assert duplicate_groups[0]['include'] == [1]
    assert duplicate_groups[0]['exclude'] == [3, 4, 5, 6]


@pytest.mark.ci
def test_database_duplicates_no_duplicates(duplicates_test_db):
    from asr.database.duplicates import main

    # Setting comparison_key = id makes removes all duplicates.
    results = main(
        connect('duplicates.db'),
        connect('duplicates_removed.db'),
        comparison_keys='id',
        filterstring='<=natoms,<id',
    )

    duplicate_groups = results['duplicate_groups']
    assert not duplicate_groups
