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
        main(uniform_vacuum=vac)
        pathlist = list(Path('.').glob('defects.BN_331*/charge_0/'))
        for path in pathlist:
            structure = read(path / 'unrelaxed.json')
            cell = structure.get_cell()
            assert cell[2, 2] == vac


@pytest.mark.ci
def test_setup_supercell(asr_tmpdir):
    from asr.setup.defects import setup_supercell
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


@pytest.mark.ci
def test_intrinsic_single_defects(asr_tmpdir):
    from pathlib import Path
    from asr.core import chdir
    from asr.setup.defects import main
    from ase.io import write
    from .materials import std_test_materials

    lengths = [1, 4, 1]
    std_test_materials.pop(2)
    for i, atoms in enumerate(std_test_materials):
        name = atoms.get_chemical_formula()
        Path(name).mkdir()
        write(f'{name}/unrelaxed.json', atoms)
        with chdir(name):
            main(general_algorithm=15.)
            pathlist = list(Path('.').glob('defects.*/charge_0'))
            assert len(pathlist) == lengths[i]


@pytest.mark.ci
def test_extrinsic_single_defects(asr_tmpdir):
    from pathlib import Path
    from asr.core import chdir
    from asr.setup.defects import main
    from ase.io import write
    from .materials import std_test_materials

    lengths = [3, 8, 3]
    std_test_materials.pop(2)
    for i, atoms in enumerate(std_test_materials):
        name = atoms.get_chemical_formula()
        Path(name).mkdir()
        write(f'{name}/unrelaxed.json', atoms)
        with chdir(name):
            main(general_algorithm=15., extrinsic='V,Nb')
            pathlist = list(Path('.').glob('defects.*/charge_0'))
            assert len(pathlist) == lengths[i]


@pytest.mark.ci
def test_extrinsic_double_defects(asr_tmpdir):
    from pathlib import Path
    from asr.core import chdir
    from asr.setup.defects import main
    from ase.io import write
    from .materials import std_test_materials

    lengths = [15]
    std_test_materials = [std_test_materials[1]]
    for i, atoms in enumerate(std_test_materials):
        name = atoms.get_chemical_formula()
        Path(name).mkdir()
        write(f'{name}/unrelaxed.json', atoms)
        with chdir(name):
            main(general_algorithm=16., extrinsic='Nb',
                 double=True)
            pathlist = list(Path('.').glob('defects.*/charge_0'))
            assert len(pathlist) == lengths[i]


@pytest.mark.ci
def test_new_double():
    from asr.setup.defects import is_new_complex

    complex_list = ['v_N.v_B', 'Cr_N.N_B', 'N_B.B_N', 'Nb_B.F_N']
    newlist = ['N_B.Cr_N', 'Cr_N.N_B', 'v_N.v_B', 'F_N.I_B', 'V_N.v_B']
    refs = [False, False, False, True, True]
    for i, new in enumerate(newlist):
        el1 = new.split('.')[0]
        el2 = new.split('.')[1]
        assert is_new_complex(el1, el2, complex_list) == refs[i]


@pytest.mark.ci
def test_setup_halfinteger(asr_tmpdir):
    from pathlib import Path
    from asr.core import chdir
    from asr.setup.defects import main
    from ase.io import write
    from .materials import std_test_materials

    atoms = std_test_materials[1]
    write('unrelaxed.json', atoms)
    main()
    p = Path('.')
    pathlist = list(p.glob('defects.*/charge_0'))
    for path in pathlist:
        with chdir(path):
            write('structure.json', atoms)
            main(halfinteger=True)
            plus = Path('sj_+0.5/params.json')
            minus = Path('sj_-0.5/params.json')
            assert plus.is_file()
            assert minus.is_file()
