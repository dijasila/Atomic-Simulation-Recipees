import pytest


@pytest.mark.ci
def test_nearest_distance():
    from asr.setup.defects import return_distances_cell
    from .materials import std_test_materials
    import numpy as np

    atoms = std_test_materials[1]
    cell = atoms.get_cell()
    distances = return_distances_cell(cell)
    refs = [cell[0][0], np.sqrt(cell[1][0]**2 + cell[1][1]**2),
            np.sqrt((-cell[0][0] + cell[1][0])**2 + cell[1][1]**2),
            np.sqrt((cell[0][0] + cell[1][0])**2 + cell[1][1]**2)]
    for i, dist in enumerate(distances):
        assert dist == refs[i]


@pytest.mark.ci
def test_setup_defects(asr_tmpdir):
    from pathlib import Path
    from .materials import std_test_materials
    from asr.setup.defects import main
    from ase.io import write, read
    from ase.calculators.calculator import compare_atoms

    atoms = std_test_materials[1]
    write('unrelaxed.json', atoms)
    atoms = atoms.repeat((3, 3, 1))
    main(supercell=(3, 3, 1))
    pristine = read('defects.pristine_sc.331/structure.json')
    assert compare_atoms(atoms, pristine) == []
    # for i in range(len(atoms.get_scaled_positions())):
    #     assert atoms.get_scaled_positions(
    #     )[i][0] == pristine.get_scaled_positions()[i][0]
    #     assert atoms.get_scaled_positions(
    #     )[i][1] == pristine.get_scaled_positions()[i][1]
    #     assert atoms.get_scaled_positions(
    #     )[i][2] == pristine.get_scaled_positions()[i][2]
    # for i in range(len(atoms.get_cell())):
    #     assert atoms.get_cell()[i][0] == pristine.get_cell()[i][0]
    #     assert atoms.get_cell()[i][1] == pristine.get_cell()[i][1]
    #     assert atoms.get_cell()[i][2] == pristine.get_cell()[i][2]
    # for i in range(len(atoms.numbers)):
    #     assert atoms.numbers[i] == pristine.numbers[i]

    pathlist = list(Path('.').glob('defects.BN_331*/charge_*/'))
    for path in pathlist:
        print(Path(path / 'params.json'))
        assert Path(path / 'params.json').is_file()
        if str(path.absolute()).endswith('charge_0'):
            assert Path(path / 'unrelaxed.json').is_file()
        else:
            assert Path(path / 'unrelaxed.json').is_symlink()

    assert Path('defects.pristine_sc.331/structure.json').is_file()


@pytest.mark.ci
def test_vacuum(asr_tmpdir):
    import numpy as np
    from pathlib import Path
    from asr.setup.defects import main
    from ase.io import read, write
    from asr.core import chdir
    from .materials import std_test_materials

    atoms = std_test_materials[1]
    for vac in np.arange(20, 30, 1):
        Path(f'{int(vac)}').mkdir()
        with chdir(f'{int(vac)}'):
            write('unrelaxed.json', atoms)
            main(general_algorithm=15., uniform_vacuum=vac)
            pathlist = list(Path('.').glob('defects.BN_*/charge_0/'))
            for path in pathlist:
                structure = read(path / 'unrelaxed.json')
                cell = structure.get_cell()
                assert cell[2, 2] == pytest.approx(vac)


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
    materials = std_test_materials.copy()
    materials.pop(2)
    for i, atoms in enumerate(materials):
        name = atoms.get_chemical_formula()
        Path(name).mkdir()
        write(f'{name}/unrelaxed.json', atoms)
        with chdir(name):
            main()
            pathlist = list(Path('.').glob('defects.*/charge_0'))
            assert len(pathlist) == lengths[i]


@pytest.mark.ci
def test_chemical_elements(asr_tmpdir):
    from pathlib import Path
    from asr.core import chdir
    from .materials import std_test_materials
    from asr.setup.defects import add_intrinsic_elements
    results = {'Si2': ['Si'],
               'BN': ['B', 'N'],
               'Ag': ['Ag'],
               'Fe': ['Fe']}
    for i, atoms in enumerate(std_test_materials):
        name = atoms.get_chemical_formula()
        Path(name).mkdir()
        with chdir(name):
            elements = add_intrinsic_elements(atoms, elements=[])
            for element in elements:
                assert element in results[name]
                assert len(elements) == len(results[name])


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
            main(extrinsic='V,Nb')
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
    from asr.setup.defects import is_new_double_defect

    complex_list = ['v_N.v_B', 'Cr_N.N_B', 'N_B.B_N', 'Nb_B.F_N']
    newlist = ['N_B.Cr_N', 'Cr_N.N_B', 'v_N.v_B', 'F_N.I_B', 'V_N.v_B']
    refs = [False, False, False, True, True]
    for i, new in enumerate(newlist):
        el1 = new.split('.')[0]
        el2 = new.split('.')[1]
        assert is_new_double_defect(el1, el2, complex_list) == refs[i]


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


@pytest.mark.ci
def test_setup_halfinteger(asr_tmpdir):
    from pathlib import Path
    from asr.core import chdir, read_json
    from asr.setup.defects import main, write_halfinteger_files
    from ase.io import write
    from .materials import std_test_materials

    materials = std_test_materials.copy()
    materials.pop(2)
    for atoms in materials:
        Path(f'{atoms.get_chemical_formula()}').mkdir()
        write(f'{atoms.get_chemical_formula()}/unrelaxed.json', atoms)
        with chdir(f'{atoms.get_chemical_formula()}'):
            main()
            p = Path('.')
            pathlist = list(p.glob('defects.*/charge_*'))
            for path in pathlist:
                with chdir(path):
                    write('structure.json', atoms)
                    params = read_json('params.json')
                    charge = int(str(path.absolute()).split('/')[-1].split('_')[-1])
                    write_halfinteger_files(0.5, '+0.5', params, charge, '.')
                    write_halfinteger_files(-0.5, '-0.5', params, charge, '.')
                    params_p = read_json('sj_+0.5/params.json')
                    params_m = read_json('sj_-0.5/params.json')
                    deltas = [0.5, -0.5]
                    for i, par in enumerate([params_p, params_m]):
                        assert par['asr.gs@calculate']['calculator']['charge'] == charge + deltas[i]
