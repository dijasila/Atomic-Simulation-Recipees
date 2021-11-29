import pytest


@pytest.mark.ci
def test_setup_defects(asr_tmpdir):
    from pathlib import Path
    from .materials import std_test_materials
    from asr.setup.defects import main
    from ase.io import write, read

    atoms = std_test_materials[1]
    write('unrelaxed.json', atoms)
    atoms = atoms.repeat((3, 3, 1))
    main(supercell=[3, 3, 1])
    pristine = read('defects.pristine_sc.331/structure.json')
    for i in range(len(atoms.get_scaled_positions())):
        assert atoms.get_scaled_positions(
        )[i][0] == pristine.get_scaled_positions()[i][0]
        assert atoms.get_scaled_positions(
        )[i][1] == pristine.get_scaled_positions()[i][1]
        assert atoms.get_scaled_positions(
        )[i][2] == pristine.get_scaled_positions()[i][2]
    for i in range(len(atoms.get_cell())):
        assert atoms.get_cell()[i][0] == pristine.get_cell()[i][0]
        assert atoms.get_cell()[i][1] == pristine.get_cell()[i][1]
        assert atoms.get_cell()[i][2] == pristine.get_cell()[i][2]
    for i in range(len(atoms.numbers)):
        assert atoms.numbers[i] == pristine.numbers[i]

    pathlist = list(Path('.').glob('defects.BN_331*/charge_*/'))
    for path in pathlist:
        assert Path(path / 'params.json').is_file()
        assert (Path(path / 'unrelaxed.json').is_file()
                or Path(path / 'unrelaxed.json').is_symlink())

    assert Path('defects.pristine_sc.331/structure.json').is_file()


@pytest.mark.ci
def test_vacuum(asr_tmpdir):
    import numpy as np
    from pathlib import Path
    from asr.setup.defects import main
    from ase.io import read, write
    from .materials import std_test_materials

    atoms = std_test_materials[1]
    write('unrelaxed.json', atoms)
    for vac in np.arange(20, 30, 1):
        main(supercell=[3, 3, 1], uniform_vacuum=vac)
        pathlist = list(Path('.').glob('defects.BN_331*/charge_0/'))
        for path in pathlist:
            structure = read(path / 'unrelaxed.json')
            cell = structure.get_cell()
            assert cell[2, 2] == vac


@pytest.mark.ci
def test_setup_supercell(asr_tmpdir):
    import numpy as np
    from pathlib import Path
    from asr.setup.defects import setup_supercell
    from ase.io import read, write
    from .materials import std_test_materials, GaAs

    atoms = [std_test_materials[1], GaAs]
    dim = [True, False]
    x = [6, 4]
    y = [6, 4]
    z = [1, 4]
    for i, atom in enumerate(atoms):
        structure, N_x, N_y, N_z = setup_supercell(atom,
                                                   15,
                                                   dim[i])
        assert N_x == x[i]
        assert N_y == y[i]
        assert N_z == z[i]
        assert len(structure) == x[i] * y[i] * z[i] * len(atom)
