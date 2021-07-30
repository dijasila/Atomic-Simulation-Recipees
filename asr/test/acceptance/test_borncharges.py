import pytest
import numpy as np


@pytest.mark.xfail
@pytest.mark.acceptance_test
def test_gpaw_berry_get_berry_phases_integration(asr_tmpdir_w_params):
    from ..materials import BN
    from asr.borncharges import main
    from asr.setup.params import main as setupparams

    calculator_params = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 300},
        'xc': 'PBE',
        'kpts': {'density': 2.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'nbands': '100%',
        'txt': 'gs.txt',
        'charge': 0,
    }

    formal_calculator = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 300},
        'xc': 'PBE',
        'kpts': {'density': 2.0},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'txt': 'formalpol.txt',
        'charge': 0
    }
    setupparams(params={'asr.gs@calculate': {'calculator': calculator_params},
                        'asr.c2db.formalpolarization':
                        {'calculator': formal_calculator}})
    BN.write('structure.json')

    results = main()

    ZB_vv = np.eye(3)
    diag = np.eye(3, dtype=bool)
    ZB_vv[diag] = [2.71, 2.71, 0.27]
    for Z_vv, sym in zip(results['Z_avv'], results['sym_a']):
        if sym == 'B':
            sign = 1
        else:
            sign = -1

        assert np.all(sign * Z_vv[diag] > 0)
        assert Z_vv[~diag] == pytest.approx(0, abs=0.001)
        assert Z_vv == pytest.approx(sign * ZB_vv, abs=0.5)
