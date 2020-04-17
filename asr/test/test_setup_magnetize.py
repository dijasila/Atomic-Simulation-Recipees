from .materials import std_test_materials, Ag2
import pytest


@pytest.mark.ci
@pytest.mark.parametrize("inputatoms", [Ag2] + std_test_materials)
def test_setup_magnetize(asr_tmpdir_w_params, inputatoms):
    import numpy as np
    from asr.core import magnetic_atoms
    from asr.setup.magnetize import main
    from ase.io import write, read
    from pathlib import Path
    write('unrelaxed.json', inputatoms)
    main(state='nm')

    assert Path('nm').is_dir()
    assert Path('nm', 'unrelaxed.json').is_file()

    atoms = read('nm/unrelaxed.json')

    assert all(atoms.get_initial_magnetic_moments() == 0.0)

    main(state='fm')
    atoms = read('fm/unrelaxed.json')
    assert all(atoms.get_initial_magnetic_moments() == 1.0)

    main(state='afm')
    magnetic = magnetic_atoms(inputatoms)
    if sum(magnetic) == 2:
        atoms = read('afm/unrelaxed.json')
        a1, a2 = np.where(magnetic)[0]
        magmoms = atoms.get_initial_magnetic_moments()
        assert magmoms[a1] == 1.0
        assert magmoms[a2] == -1.0


@pytest.mark.ci
@pytest.mark.parametrize("inputatoms", [Ag2])
@pytest.mark.parametrize("state", ['nm', 'fm', 'afm', 'nm,fm'])
def test_setup_magnetize_state_inputs(asr_tmpdir_w_params, inputatoms,
                                      state):
    from asr.setup.magnetize import main
    from pathlib import Path
    inputatoms.write('unrelaxed.json')
    main(state=state)

    states = state.split(',')
    for directory in states:
        assert Path(directory).is_dir()
