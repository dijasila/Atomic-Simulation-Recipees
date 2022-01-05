import pytest
from .materials import BN


@pytest.mark.ci
def test_defect_symmetry(asr_tmpdir):
    from pathlib import Path
    from ase.io import write
    from asr.core import chdir
    from asr.gs import calculate, main
    from asr.get_wfs import main as get_wfs
    from asr.defect_symmetry import main as defect_symmetry
    from asr.setup.defects import main as setup_defects

    write('unrelaxed.json', BN)
    setup_defects(supercell=[3, 3, 1])

    p = Path('.')
    pathlist = list(p.glob('defects.BN_331.v_N/charge_0/'))

    with chdir(Path(p / 'defects.pristine_sc.331')):
        calculate(
            calculator={
                "name": "gpaw",
                "mode": "lcao",
                "kpts": {"density": 1, "gamma": True},
            },
        )
        main()

    for path in pathlist:
        with chdir(path):
            write('structure.json', BN)
            calculate(
                calculator={
                    "name": "gpaw",
                    "mode": "lcao",
                    "kpts": {"density": 1, "gamma": True},
                },
            )
            main()
            get_wfs()
            symresults = defect_symmetry()

            defectname = str(path.absolute()).split(
                'BN_331.')[-1].split('/charge_0')[0]
            assert symresults['defect_name'] == defectname
            assert symresults['pointgroup'] == 'D3h'
            for element in symresults['symmetries']:
                assert element['best'] == "A'1"


@pytest.mark.ci
def test_defect_mapping(asr_tmpdir):
    from pathlib import Path
    from ase.io import write, read
    from asr.core import chdir
    from asr.defect_symmetry import (check_and_return_input,
                                     get_mapped_structure,
                                     get_spg_symmetry,
                                     get_defect_info,
                                     return_defect_coordinates)
    from asr.setup.defects import main as setup_defects

    write('unrelaxed.json', BN)
    setup_defects(general_algorithm=15.)

    p = Path('.')
    pathlist = list(p.glob('defects.BN_000.v_N/charge_0/'))

    for path in pathlist:
        with chdir(path):
            atoms = read('unrelaxed.json')
            write('structure.json', atoms)
            defect = Path('.')
            structure, unrelaxed, primitive, pristine = check_and_return_input()
            mapped_structure = get_mapped_structure(structure,
                                                    unrelaxed,
                                                    primitive,
                                                    pristine,
                                                    defect)
            point_group = get_spg_symmetry(mapped_structure)
            defecttype, defectpos = get_defect_info(primitive, defect)
            defectname = defecttype + '_' + defectpos
            center = return_defect_coordinates(structure,
                                               unrelaxed,
                                               primitive,
                                               pristine,
                                               defect)

            refname = str(path.absolute()).split(
                'BN_000.')[-1].split('/charge_0')[0]
            assert defectname == refname
            assert point_group == 'D3h'
            assert center[0] == pytest.approx(15.0599999999999)
            assert center[1] == pytest.approx(1.44914917566596)
            assert center[2] == pytest.approx(7.5)
