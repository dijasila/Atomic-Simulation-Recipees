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
def test_setup_defects(repo):
    from pathlib import Path
    from .materials import std_test_materials
    from asr.setup.defects import main
    from ase.calculators.calculator import compare_atoms
    from ase.io import read

    pathname = 'asr.setup.defects.defect*/'
    atoms = std_test_materials[1]
    repo.run_workflow_blocking(main, atoms=atoms, supercell=(3, 3, 1))
    atoms = atoms.repeat((3, 3, 1))
    pathlist = list(Path('.').glob('tree/defects.pristine_sc.331/' + pathname))
    assert len(pathlist) == 1
    pristine = read(str(pathlist[0]) + '/unrelaxed.json')
    assert compare_atoms(atoms, pristine) == []
    assert Path(pathlist[0] / 'input.json').is_file()
    assert Path(pathlist[0] / 'output.json').is_file()

    pathlist = list(Path('.').glob('tree/defects.BN_331*/' + pathname))
    for path in pathlist:
        assert Path(path / 'unrelaxed.json').is_file()
        assert Path(path / 'input.json').is_file()
        assert Path(path / 'output.json').is_file()


@pytest.mark.ci
@pytest.mark.parametrize('vac', [True, False])
def test_apply_vacuum(repo, vac):
    from pathlib import Path
    from asr.setup.defects import main
    from ase.io import read, write
    from .materials import std_test_materials
    atoms = std_test_materials[1]
    write('unrelaxed.json', atoms)
    repo.run_workflow_blocking(main, atoms=atoms, general_algorithm=15.,
                               uniform_vacuum=vac)
    pathlist = list(Path('.').glob('../*'))
    pathlist = list(Path('.').glob('tree/defects.BN_*/asr.setup.defects.defect*'))
    assert len(pathlist) > 0
    for path in pathlist:
        structure = read(path / 'unrelaxed.json')
        cell = structure.get_cell()
        ref = (cell.lengths()[0] + cell.lengths()[1]) / 2.
        if vac:
            assert cell[2, 2] == pytest.approx(ref)
        else:
            assert cell[2, 2] == pytest.approx(
                atoms.get_cell().lengths()[2])


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
def test_intrinsic_single_defects(repo):
    from pathlib import Path
    from asr.setup.defects import main
    from .materials import std_test_materials

    lengths = [1, 4, 1]
    materials = std_test_materials.copy()
    materials.pop(2)
    for i, atoms in enumerate(materials):
        repo.run_workflow_blocking(main, atoms=atoms)
        name = atoms.get_chemical_formula()
        pathlist = list(Path('./tree').glob('defects.'
                        + name + '_*/asr.setup.defects.defect*'))
        assert len(pathlist) == lengths[i]


@pytest.mark.ci
def test_chemical_elements(asr_tmpdir):
    from .materials import std_test_materials
    from asr.setup.defects import add_intrinsic_elements
    results = {'Si2': ['Si'],
               'BN': ['B', 'N'],
               'Ag': ['Ag'],
               'Fe': ['Fe']}
    for i, atoms in enumerate(std_test_materials):
        name = atoms.get_chemical_formula()
        elements = add_intrinsic_elements(atoms, elements=[])
        for element in elements:
            assert element in results[name]
            assert len(elements) == len(results[name])


@pytest.mark.ci
def test_extrinsic_single_defects(repo):
    from pathlib import Path
    from asr.setup.defects import main
    from .materials import std_test_materials

    lengths = [3, 8, 3]
    std_test_materials.pop(2)
    for i, atoms in enumerate(std_test_materials):
        repo.run_workflow_blocking(main, atoms=atoms, extrinsic='V,Nb')
        name = atoms.get_chemical_formula()
        pathlist = list(Path('./tree').glob('defects.'
                        + name + '_*/asr.setup.defects.defect*'))
        assert len(pathlist) == lengths[i]


@pytest.mark.parametrize('double_type', ['vac-vac',
                                         'vac-sub',
                                         'sub-sub'])
@pytest.mark.ci
def test_extrinsic_double_defects(double_type, repo):
    from pathlib import Path
    from asr.setup.defects import main
    from .materials import std_test_materials

    lengths = {'vac-vac': 8,
               'vac-sub': 12,
               'sub-sub': 13}
    std_test_materials = [std_test_materials[1]]
    for i, atoms in enumerate(std_test_materials):
        repo.run_workflow_blocking(main, atoms=atoms, extrinsic='Nb',
                                   double=double_type, scaling_double=1.5)
        name = atoms.get_chemical_formula()
        pathlist = list(Path('./tree').glob('defects.' + name
                        + '_*/asr.setup.defects.defect*'))
        assert len(pathlist) == lengths[double_type]


@pytest.mark.parametrize('double_type', ['vac-vac',
                                         'vac-sub',
                                         'sub-sub'])
@pytest.mark.ci
def test_exclude_double_defects(double_type, repo):
    from pathlib import Path
    from asr.setup.defects import main
    from .materials import std_test_materials
    lengths = {'vac-vac': 10,
               'vac-sub': 17,
               'sub-sub': 21}
    std_test_materials = [std_test_materials[1]]
    for i, atoms in enumerate(std_test_materials):
        repo.run_workflow_blocking(main, atoms=atoms, extrinsic='Nb,Yb',
                                   double=double_type, double_exclude='Yb',
                                   scaling_double=1.5)
        name = atoms.get_chemical_formula()
        pathlist = list(Path('./tree').glob('defects.' + name +
                        '_*/asr.setup.defects.defect*'))
        assert len(pathlist) == lengths[double_type]


@pytest.mark.parametrize('M', ['Mo', 'W'])
@pytest.mark.parametrize('X', ['S', 'Se', 'Te'])
@pytest.mark.parametrize('scaling', [0, 1, 1.5, 2])
@pytest.mark.ci
def test_get_maximum_distance(M, X, scaling):
    from asr.setup.defects import get_maximum_distance
    from ase.data import atomic_numbers, covalent_radii
    from ase.build import mx2

    atoms = mx2(f'{M}{X}2')
    metal = atomic_numbers[M]
    chalcogen = atomic_numbers[X]
    reference = ((covalent_radii[metal] + covalent_radii[chalcogen])
                 * scaling)

    R = get_maximum_distance(atoms, 0, 1, scaling)
    assert R == pytest.approx(reference)


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


@pytest.mark.xfail
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


@pytest.mark.xfail
@pytest.mark.ci
def test_write_halfinteger(asr_tmpdir):
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
                        assert (par['asr.gs@calculate']['calculator']['charge']
                                == charge + deltas[i])
