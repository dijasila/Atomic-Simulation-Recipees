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
