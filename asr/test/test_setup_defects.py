import pytest

@pytest.mark.ci
def test_setup_defects_specify_supercell(asr_tmpdir):
    from .materials import std_test_materials
    from asr.setup.defects import main
    from ase.io import write, read
    atoms = std_test_materials[1]
    write('unrelaxed.json', atoms)
    atoms = atoms.repeat((3, 3, 1))
    main(supercell=[3, 3, 1], vacuum=15.0)
    pristine = read('defects.pristine_sc/structure.json')
    for i in range(len(atoms.get_scaled_positions())):
        assert atoms.get_scaled_positions()[i][0] == pristine.get_scaled_positions()[i][0]
        assert atoms.get_scaled_positions()[i][1] == pristine.get_scaled_positions()[i][1]
        assert atoms.get_scaled_positions()[i][2] == pristine.get_scaled_positions()[i][2]
