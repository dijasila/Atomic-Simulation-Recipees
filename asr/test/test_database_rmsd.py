import pytest
from ase import Atoms
from ase.db import connect


@pytest.mark.ci
def test_database_rmsd_duplicates(duplicates_test_db):
    """Test that the duplicates (with rmsd=0)are correctly identified."""
    from asr.database.rmsd import main

    nmat = len(duplicates_test_db)
    results = main(
        connect('duplicates.db'), connect('duplicates-rmsd.db'))
    rmsd_by_id = results['rmsd_by_id']
    assert set(range(1, nmat + 1)).issubset(set(rmsd_by_id.keys()))
    for i in range(1, nmat + 1):
        keys = set([j for j in range(1, nmat + 1) if j != i])
        assert keys.issubset(set(rmsd_by_id[i].keys()))

        for j in keys:
            assert rmsd_by_id[i][j] == pytest.approx(0)


@pytest.mark.ci
def test_database_rmsd_duplicates_comparison_keys(duplicates_test_db):
    """Test that the duplicates (with rmsd=0)are correctly identified."""
    from asr.database.rmsd import main

    results = main(connect('duplicates.db'), connect('duplicates-rmsd.db'),
                   comparison_keys='magstate')
    rmsd_by_id = results['rmsd_by_id']
    assert set(rmsd_by_id.keys()) == set([1, 3, 4, 5, 6])


@pytest.mark.ci
@pytest.mark.parametrize('angle', [30])
@pytest.mark.parametrize('vector', ['x', 'y', 'z'])
def test_database_rmsd_rotation(test_material,
                                angle, vector):
    """Test that rmsd=0 when comparing rotated structures."""
    from asr.database.rmsd import get_rmsd

    atoms = test_material.copy()
    atoms.rotate(angle, v=vector, rotate_cell=True)
    rmsd = get_rmsd(test_material, atoms)
    assert rmsd == pytest.approx(0)


@pytest.mark.ci
def test_database_rmsd_none_handling(asr_tmpdir):
    """Test that handling of "None" from get_rmsd is correct."""
    from .materials import Si
    from ase.db import connect
    from asr.database.rmsd import main
    db = connect('duplicates_rattled.db')

    db.write(Si)
    rattled = Si.copy()
    rattled.rattle(1.0)
    db.write(rattled)
    main(
        connect('duplicates_rattled.db'),
        connect('duplicates_rattled_rmsd.db'), max_rmsd=0.001)


def rattle_atoms(atoms, scale=0.01, seed=42):
    import numpy as np
    rng = np.random.RandomState(seed)
    pos = atoms.get_positions()
    dir_av = rng.normal(scale=scale, size=pos.shape)
    dir_av /= np.linalg.norm(dir_av, axis=1)[:, None]
    atoms.set_positions(pos + dir_av * scale)
    return atoms


@pytest.mark.ci
def test_database_rmsd_rattled(test_material):
    """Test that rattled structures have a finite rmsd."""
    import numpy as np
    from asr.database.rmsd import get_rmsd

    pbc_c = test_material.get_pbc()
    repeat = np.array([3, 3, 3], int)
    repeat[~pbc_c] = 1
    rattled_atoms = test_material.repeat(repeat)
    rattle_atoms(rattled_atoms, 0.01, seed=42)

    rmsd = get_rmsd(test_material, rattled_atoms)
    assert rmsd > 0.0, (test_material.repeat(repeat).get_scaled_positions()
                        - rattled_atoms.get_scaled_positions())


@pytest.mark.ci
@pytest.mark.parametrize('atom_index', [0, -1])
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('displacement', [0.01, 0.02])
def test_database_rmsd_move_one_atom(test_material,
                                     atom_index, axis, displacement):
    """Test that rmsd=0 when comparing rotated structures."""
    import numpy as np
    from asr.database.rmsd import get_rmsd

    pbc_c = test_material.get_pbc()
    repeat = np.array([1, 1, 1], int)
    repeat[np.argwhere(pbc_c)[0, 0]] = 2
    atoms = test_material.repeat(repeat)
    translations_av = np.zeros((len(atoms), 3), float)
    translations_av[atom_index, axis] = displacement
    atoms.translate(translations_av)
    rmsd = get_rmsd(test_material, atoms)
    # Logic of the formula below:
    # The reference stucture is displaced according to the average displacement
    # of all atoms hence the second term.
    disp = displacement - displacement / len(atoms)
    assert rmsd == pytest.approx(disp)


@pytest.mark.ci
@pytest.mark.parametrize('atoms1,atoms2', [
    (
        Atoms(symbols='Co2S2',
              pbc=[True, True, False],
              cell=[[3.5790788191969725, -1.1842760125086163e-20, 0.0],
                    [-1.7895394075540594, 3.10048672285293, 0.0],
                    [2.3583795244967227e-18, 0.0, 18.85580293064]],
              scaled_positions=[[0, 0, 0.56],
                                [1 / 3, 2 / 3, 0.44],
                                [1 / 3, 2 / 3, 0.40],
                                [0, 0, 0.60]]),
        Atoms(symbols='Co2S2',
              pbc=[True, True, False],
              cell=[[3.5790788191969725, -1.1842760125086163e-20, 0.0],
                    [-1.7895394075540594, 3.10048672285293, 0.0],
                    [2.3583795244967227e-18, 0.0, 18.85580293064]],
              scaled_positions=[[0, 0, 0.56],
                                [0, 0, 0.44],
                                [1 / 3, 2 / 3, 0.40],
                                [2 / 3, 1 / 3, 0.60]])
    )
])
def test_database_rmsd_not_equal(atoms1, atoms2):
    """Test some explicit cases that have previously posed a problem."""
    from asr.database.rmsd import get_rmsd
    rmsd = get_rmsd(atoms1, atoms2)
    assert not rmsd < 0.5
