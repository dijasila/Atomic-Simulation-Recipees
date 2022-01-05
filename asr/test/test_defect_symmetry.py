import pytest


@pytest.mark.parametrize('extrinsic', ['NO', 'C', 'Se,Te'])
@pytest.mark.parametrize('intrinsic', [True, False])
@pytest.mark.parametrize('vacancies', [True, False])
@pytest.mark.ci
def test_get_defect_info(asr_tmpdir, extrinsic, intrinsic, vacancies):
    from .materials import BN
    from pathlib import Path
    from ase.io import write
    from asr.defect_symmetry import get_defect_info, is_vacancy
    from asr.setup.defects import main

    atoms = BN.copy()
    write('unrelaxed.json', atoms)
    main(extrinsic=extrinsic,
         intrinsic=intrinsic,
         vacancies=vacancies)
    p = Path('.')
    pathlist = list(p.glob('defects.BN*/charge_0'))
    for path in pathlist:
        defecttype, defectpos = get_defect_info(path)
        string = str(path.absolute()).split('/')[-2].split('.')[-1]
        assert defecttype == string.split('_')[0]
        assert defectpos == string.split('_')[1]
        if string.split('_')[0] == 'v':
            assert is_vacancy(path)


@pytest.mark.ci
def test_get_supercell_shape(asr_tmpdir):
    from materials import BN
    from asr.defect_symmetry import get_supercell_shape

    atoms = BN.copy()
    for i in range(1, 10):
        for j in range(1, 10):
            pristine = atoms.repeat((i, j, 1))
            N = get_supercell_shape(atoms, pristine)
            assert N == min(i, j)


@pytest.mark.parametrize('is_vacancy', [True, False])
@pytest.mark.ci
def test_conserved_atoms(is_vacancy):
    from asr.defect_symmetry import conserved_atoms
    from .materials import BN

    atoms = BN.copy()
    for i in range(2, 10):
        for j in range(len(atoms) * i):
            supercell = atoms.repeat((i, i, 1))
            if is_vacancy:
                supercell.pop(j)
                print(len(supercell), len(atoms), i)
                assert conserved_atoms(supercell,
                                       atoms,
                                       i,
                                       is_vacancy)
            else:
                supercell.symbols[j] = 'X'
                assert conserved_atoms(supercell,
                                       atoms,
                                       i,
                                       is_vacancy)
