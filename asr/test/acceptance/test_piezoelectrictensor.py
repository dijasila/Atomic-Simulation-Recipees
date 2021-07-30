import numpy as np
import pytest


@pytest.mark.xfail
@pytest.mark.acceptance_test
def test_piezo_BN(asr_tmpdir_w_params):
    from ..materials import BN
    from asr.c2db.piezoelectrictensor import main
    from asr.setup.params import main as setupparams

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
    setupparams(params={'asr.c2db.formalpolarization':
                        {'calculator': formal_calculator}})
    BN.write('structure.json')
    results = main()

    full_response = 0.12  # From a default parameter calculation of BN
    clamped_response = 0.044  # From a default parameter calculation of BN
    for eps, response in [('eps_vvv', full_response),
                          ('eps_clamped_vvv', clamped_response)]:
        eps_calculated_vvv = results[eps]
        eps_converged_vvv = np.array([[[0.0, response, 0.0],
                                       [response, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]],
                                      [[response, 0.0, 0.0],
                                       [0.0, -response, 0.0],
                                       [0.0, 0.0, 0.0]],
                                      [[0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]]])

        assert eps_calculated_vvv == pytest.approx(eps_converged_vvv, rel=0.05)
