import pytest
from pytest import approx

@pytest.mark.ci
def test_setup_defects_supercell():
    #asr_tmpdir
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
    for i in range(len(atoms.get_cell())):
        assert atoms.get_cell()[i][0] == pristine.get_cell()[i][0]
        assert atoms.get_cell()[i][1] == pristine.get_cell()[i][1]
        assert atoms.get_cell()[i][2] == pristine.get_cell()[i][2]
    for i in range(len(atoms.numbers)):
        assert atoms.numbers[i] == pristine.numbers[i]

    defect_ref = atoms.copy()
    defect_ref.pop(0)
    defect_rec = read('defects.BN_331.v_B/charge_0/unrelaxed.json')
    for i in range(len(defect_ref.get_scaled_positions())):
        assert defect_ref.get_scaled_positions()[i][0] == approx(defect_rec.get_scaled_positions()[i][0], abs=1e-2)
        assert defect_ref.get_scaled_positions()[i][1] == approx(defect_rec.get_scaled_positions()[i][1], abs=1e-2)
        assert defect_ref.get_scaled_positions()[i][2] == approx(defect_rec.get_scaled_positions()[i][2], abs=1e-2)
    for i in range(len(defect_ref.get_cell())):
        assert defect_ref.get_cell()[i][0] == approx(defect_rec.get_cell()[i][0], abs=1e-2)
        assert defect_ref.get_cell()[i][1] == approx(defect_rec.get_cell()[i][1], abs=1e-2)
        assert defect_ref.get_cell()[i][2] == approx(defect_rec.get_cell()[i][2], abs=1e-2)
    for i in range(len(defect_ref.numbers)):
        assert defect_ref.numbers[i] == defect_rec.numbers[i]
