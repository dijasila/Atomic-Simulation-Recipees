from .materials import BN, GaAs, Si
import pytest


@pytest.mark.ci
@pytest.mark.parametrize("inputatoms", [Si, BN, GaAs])
def test_shift(asr_tmpdir_w_params, inputatoms, mockgpaw, mocker, get_webcontent):
    from asr.shg import get_chi_symmetry, CentroSymmetric
    from asr.shift import main
    import numpy as np
    import gpaw
    import gpaw.nlopt.shift

    print(inputatoms.get_chemical_symbols())

    sym_chi = get_chi_symmetry(inputatoms, sym_th=1e-3)
    comp = ''
    for rel in sym_chi.values():
        comp += '=' + rel
    comp = comp[1:]
    comp = comp.split('=')
    assert len(comp) == 27, 'Error in get_chi_symmetry'

    w_ls = np.array([0.0, 1.0, 2.0, 3.0])

    def get_shift(
            freqs=w_ls, **kargw):

        chi = np.random.rand(len(freqs)) + 1j * np.random.random(len(freqs))
        return np.vstack((freqs, chi))

    mocker.patch.object(gpaw.nlopt.shift, 'get_shift', get_shift)

    # Check the main function and webpanel
    if inputatoms.get_chemical_symbols()[0] == 'Si':
        with pytest.raises(CentroSymmetric):
            assert main(atoms=inputatoms, maxomega=3, nromega=4)
    else:
        main(atoms=inputatoms, maxomega=3, nromega=4)
        inputatoms.write('structure.json')
        content = get_webcontent()
        assert 'shift' in content
