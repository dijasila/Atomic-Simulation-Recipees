from .materials import std_test_materials, Ag2
import pytest


@pytest.mark.ci
@pytest.mark.parametrize("inputatoms", [Ag2] + std_test_materials)
def test_setup_magnetize(asr_tmpdir_w_params, inputatoms):
    import numpy as np
    from asr.utils import magnetic_atoms
    from asr.setup.magnetize import main
    atoms = main(atoms=inputatoms, state='nm')

    assert all(atoms.get_initial_magnetic_moments() == 0.0)

    atoms = main(atoms=inputatoms, state='fm')
    assert all(atoms.get_initial_magnetic_moments() == 1.0)

    magnetic = magnetic_atoms(inputatoms)
    if sum(magnetic) == 2:
        atoms = main(atoms=inputatoms, state='afm')
        a1, a2 = np.where(magnetic)[0]
        magmoms = atoms.get_initial_magnetic_moments()
        assert magmoms[a1] == 1.0
        assert magmoms[a2] == -1.0
