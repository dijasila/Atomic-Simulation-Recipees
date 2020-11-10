
import pytest


@pytest.mark.ci
def test_shg(asr_tmpdir_w_params, test_material, mockgpaw, mocker, get_webcontent):
    from asr.shg import get_chi_symmtery, make_full_chi, calc_polarized_shg, main
    from ase.io import read
    from asr.core import write_file, ASRResult
    import numpy as np
    # import gpaw
    # import gpaw.nlopt.shg
    # import gpaw.nlopt.matrixel

    test_material.write('structure.json')
    atoms = read('structure.json')
    print(atoms.get_chemical_symbols())

    sym_chi = get_chi_symmtery(atoms, sym_th=1e-3)
    comp = ''
    for rel in sym_chi.values():
        comp += '=' + rel
    comp = comp[1:]
    comp = comp.split('=')
    assert len(comp) == 27, 'Error in get_chi_symmtery'

    w_ls = np.array([0, 1, 2, 3])
    chi_dict = {}
    for pol in sorted(sym_chi.keys()):
        if pol == 'zero':
            continue
        chi_dict[pol] = np.random.rand(len(w_ls)) + 1j * np.random.random(len(w_ls))
    result = ASRResult(
        data={
            'chi': chi_dict,
            'symm': sym_chi,
            'freqs': w_ls,
            'par': {'eta': 0.05, 'gauge': 'lg',
                    'nbands': f'{2*100}%',
                    'kpts': {'density': 10, 'gamma': True}}},
        metadata={'asr_name': 'asr.shg'})
    write_file('results-asr.shg.json', result.format_as('json'))

    if len(sym_chi) != 1:
        chi_vvvl = make_full_chi(sym_chi, chi_dict)
        assert chi_vvvl.shape == (3, 3, 3, len(w_ls))
        calc_polarized_shg(sym_chi, chi_dict)

    # def get_shg(
    #         freqs=w_ls,
    #         eta=0.05,
    #         pol='yyy',
    #         eshift=0.0,
    #         gauge='lg',
    #         ftol=1e-4, Etol=1e-6,
    #         band_n=None,
    #         out_name='shg.npy',
    #         mml_name='mml.npz'):

    #     chi = np.random.rand(len(freqs)) + 1j * np.random.random(len(freqs))
    #     return np.vstack((freqs, chi))

    # def make_nlodata(
    #         gs_name='gs.gpw', out_name='mml.npz', ni=0, nf=0):

    #     pass

    # mocker.patch.object(gpaw.nlopt.shg, 'get_shg', get_shg)
    # mocker.patch.object(gpaw.nlopt.matrixel, 'make_nlodata', make_nlodata)

    # main()

    # Check the webpanel
    content = get_webcontent()
    assert 'shg' in content
