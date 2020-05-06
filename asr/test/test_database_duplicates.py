import pytest
from .materials import std_test_materials
from ase import Atoms


@pytest.fixture(params=std_test_materials)
def duplicates_test_db(request, asr_tmpdir):
    """Set up a database containing only duplicates of BN."""
    import numpy as np
    import ase.db

    db = ase.db.connect("duplicates.db")
    atoms = request.param.copy()

    db.write(atoms=atoms)

    rotated_atoms = atoms.copy()
    rotated_atoms.rotate(23, v='z', rotate_cell=True)
    db.write(atoms=rotated_atoms, magstate='FM')

    pbc_c = atoms.get_pbc()
    repeat = np.array([2, 2, 2], int)
    repeat[~pbc_c] = 1
    supercell_ref = atoms.repeat(repeat)
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
    stretch_nonpbc_atoms.set_cell(cell)
    db.write(stretch_nonpbc_atoms)

    return (atoms, db)


@pytest.mark.ci
@pytest.mark.parametrize('angle', [30])
@pytest.mark.parametrize('vector', ['x', 'y', 'z'])
def test_database_duplicates_rmsd_duplicate(test_material,
                                            angle, vector):
    from asr.duplicates import get_rmsd

    atoms = test_material.copy()
    atoms.rotate(angle, v=vector, rotate_cell=True)
    rmsd = get_rmsd(test_material, atoms)
    assert rmsd == pytest.approx(0)


@pytest.mark.ci
def test_database_duplicates(duplicates_test_db):
    from asr.database.duplicates import main

    results = main('duplicates.db', 'duplicates_removed.db')

    nduplicates = len(duplicates_test_db[1])
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
    import numpy as np
    from asr.setup.symmetrize import atomstospgcell as ats
    import ase.db

    atoms = request.param.copy()
    pbc_c = atoms.get_pbc()
    repeat = np.array([2, 2, 2], int)
    repeat[~pbc_c] = 1
    atoms1 = atoms.repeat(repeat)
    atoms1.rattle(0.01, seed=41)
    atoms2 = atoms.repeat(repeat)
    atoms2.rattle(0.01, seed=42)
    dataset1 = spglib.get_symmetry_dataset(ats(atoms1),
                                           symprec=1e-2)
    number1 = dataset1['number']

    dataset2 = spglib.get_symmetry_dataset(ats(atoms2),
                                           symprec=1e-2)
    number2 = dataset2['number']
    assert number1 == number2

    db = ase.db.connect('very_rattled.db')
    db.write(atoms1)
    db.write(atoms2)

    return (atoms, db)


@pytest.mark.ci
def test_database_duplicates_rattled(test_material):
    import numpy as np
    from asr.duplicates import get_rmsd

    pbc_c = test_material.get_pbc()
    repeat = np.array([2, 2, 2], int)
    repeat[~pbc_c] = 1
    atoms = test_material.repeat(repeat)
    rattled_atoms = atoms.copy()
    rattled_atoms.rattle(0.01, seed=32)

    rmsd = get_rmsd(atoms, rattled_atoms)
    assert rmsd > 0.001


# @pytest.mark.ci
# @pytest.mark.parametrize('atoms1,atoms2', [
#     (
#         Atoms(symbols='Co2S2',
#               pbc=[True, True, False],
#               cell=[[3.5790788191969725, -1.1842760125086163e-20, 0.0],
#                     [-1.7895394075540594, 3.10048672285293, 0.0],
#                     [2.3583795244967227e-18, 0.0, 18.85580293064]],
#               scaled_positions=[[0, 0, 0.56],
#                                 [1 / 3, 2 / 3, 0.44],
#                                 [1 / 3, 2 / 3, 0.40],
#                                 [0, 0, 0.60]]),
#         Atoms(symbols='Co2S2',
#               pbc=[True, True, False],
#               cell=[[3.5790788191969725, -1.1842760125086163e-20, 0.0],
#                     [-1.7895394075540594, 3.10048672285293, 0.0],
#                     [2.3583795244967227e-18, 0.0, 18.85580293064]],
#               scaled_positions=[[0, 0, 0.56],
#                                 [0, 0, 0.44],
#                                 [1 / 3, 2 / 3, 0.40],
#                                 [2 / 3, 1 / 3, 0.60]])
#     )
# ])
# def test_database_duplicates_not_equal(atoms1, atoms2):
#     """Test some explicit cases that have previously posed a problem."""
#     from asr.duplicates import are_structures_duplicates
#     # from ase.visualize import view
#     print(atoms1.get_scaled_positions())
#     print(atoms2.get_scaled_positions())

#     # view(atoms1)
#     # view(atoms2)
#     assert not are_structures_duplicates(atoms1, atoms2, symprec=0.1)
