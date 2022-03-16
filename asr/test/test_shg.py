from .materials import BN, GaAs, Si
import pytest


@pytest.mark.ci
@pytest.mark.parametrize("inputatoms", [Si, BN, GaAs])
def test_shg(asr_tmpdir_w_params, inputatoms, mockgpaw, mocker, get_webcontent):

    from asr.shg import get_chi_symmetry, main, CentroSymmetric
    from ase.io import read
    import numpy as np
    import gpaw
    import gpaw.nlopt.shg

    inputatoms.write('structure.json')
    atoms = read('structure.json')
    # print(atoms.get_chemical_symbols())

    sym_chi = get_chi_symmetry(atoms, sym_th=1e-3)
    comp = ''
    for rel in sym_chi.values():
        comp += '=' + rel
    comp = comp[1:]
    comp = comp.split('=')
    assert len(comp) == 27, 'Error in get_chi_symmetry'

    w_ls = np.array([0.0, 1.0, 2.0, 3.0])

    rng = np.random.RandomState(123)

    def get_shg(
            freqs=w_ls, **kargw):

        chi = rng.random(len(freqs)) + 1j * rng.random(len(freqs))
        return np.vstack((freqs, chi))

    mocker.patch.object(gpaw.nlopt.shg, 'get_shg', get_shg)

    # Check the main function and webpanel
    if atoms.get_chemical_symbols()[0] == 'Si':
        with pytest.raises(CentroSymmetric):
            assert main(maxomega=3, nromega=4)
    else:
        main(maxomega=3, nromega=4)
        content = get_webcontent()
        assert 'shg' in content
