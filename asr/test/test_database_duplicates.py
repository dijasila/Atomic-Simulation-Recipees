import pytest
from .materials import std_test_materials


@pytest.fixture(params=std_test_materials)
def duplicates_test_db(request, asr_tmpdir):
    """Set up a database containing only duplicates of BN."""
    import ase.db

    db = ase.db.connect("duplicates.db")
    atoms = request.param.copy()

    db.write(atoms=atoms)

    rotated_atoms = atoms.copy()
    rotated_atoms.rotate(23, v='z', rotate_cell=True)
    db.write(atoms=rotated_atoms, magstate='FM')

    supercell_ref = atoms.repeat((2, 2, 2))
    db.write(supercell_ref)

    translated_atoms = atoms.copy()
    translated_atoms.translate(0.5)
    db.write(translated_atoms)

    rattled_atoms = atoms.copy()
    rattled_atoms.rattle(0.001)
    db.write(rattled_atoms)

    stretch_nonpbc_atoms = atoms.copy()
    cell = stretch_nonpbc_atoms.get_cell()
    pbc_c = atoms.get_pbc()
    cell[~pbc_c][:, ~pbc_c] *= 2
    db.write(stretch_nonpbc_atoms)

    return db


@pytest.mark.ci
def test_database_duplicates(duplicates_test_db):
    from asr.database.duplicates import main

    results = main('duplicates.db', 'duplicates_removed.db')

    nduplicates = len(duplicates_test_db)
    duplicate_groups = results['duplicate_groups']
    assert duplicate_groups[1] == list(range(1, nduplicates + 1))


@pytest.mark.ci
def test_database_duplicates_filter_magstate(duplicates_test_db):
    from asr.database.duplicates import main

    results = main('duplicates.db', 'duplicates_removed.db',
                   comparison_keys='magstate')

    duplicate_groups = results['duplicate_groups']
    assert duplicate_groups[1] == [1, 3, 4, 5, 6]


@pytest.mark.ci
def test_database_duplicates_no_duplicates(duplicates_test_db):
    from asr.database.duplicates import main

    # Setting comparison_key = id makes removes all duplicates.
    results = main('duplicates.db', 'duplicates_removed.db',
                   comparison_keys='id')

    duplicate_groups = results['duplicate_groups']
    assert not duplicate_groups


@pytest.fixture(params=std_test_materials)
def rattled_atoms_db(request, asr_tmpdir):
    import spglib
    from asr.setup.symmetrize import atomstospgcell as ats
    import ase.db

    atoms = request.param.copy()
    atoms1 = atoms.repeat((2, 2, 2))
    atoms1.rattle(0.01)
    atoms2 = atoms.repeat((2, 2, 2))
    atoms2.rattle(0.01)
    dataset1 = spglib.get_symmetry_dataset(ats(atoms1),
                                           symprec=1e-2)
    number = dataset1['number']
    assert number == 1

    dataset2 = spglib.get_symmetry_dataset(ats(atoms2),
                                           symprec=1e-2)
    number = dataset2['number']
    assert number == 1

    db = ase.db.connect('very_rattled.db')
    db.write(atoms1)
    db.write(atoms2)

    return db


@pytest.mark.ci
def test_database_duplicates_rattled_BN(rattled_atoms_db):
    from asr.database.duplicates import main

    results = main('very_rattled.db',
                   'very_rattled_no_duplicates.db',
                   comparison_keys='id')

    duplicate_groups = results['duplicate_groups']
    assert not duplicate_groups
