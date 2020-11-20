import pytest


@pytest.mark.ci
def test_ramanpol(
    asr_tmpdir_w_params, asr_tmpdir, mockgpaw, mocker,
        test_material, get_webcontent):
    from asr.core import write_file, ASRResult
    from asr.ramanpol import main
    import numpy as np
    import gpaw
    import gpaw.nlopt.matrixel
    import gpaw.nlopt.basic

    test_material.write('structure.json')

    def make_nlodata(gs_name='gs.gpw', out_name='mml.npz'):
        nk = 5
        nb = 6
        w_k = np.random.rand(nk)
        f_kn = np.random.rand(nk, nb)
        E_kn = 5 * np.random.rand(nk, nb)
        p_kvnn = (np.random.rand(nk, 3, nb, nb)
                  + 1j * np.random.rand(nk, 3, nb, nb))
        np.savez(out_name, w_k=w_k,
                 f_kn=f_kn, E_kn=E_kn, p_kvnn=p_kvnn)

    def load_data(mml_name='mml.npz'):
        nk = 5
        nlo = np.load(mml_name)
        k_info = {}
        for ii in range(nk):
            k_info[ii] = [nlo['w_k'][ii], nlo['f_kn'][ii],
                          nlo['E_kn'][ii], nlo['p_kvnn'][ii]]
        return k_info

    mocker.patch.object(gpaw.nlopt.matrixel, 'make_nlodata', make_nlodata)
    mocker.patch.object(gpaw.nlopt.basic, 'load_data', load_data)
    main(dftd3=False)

    wl = np.array([488.0, 532.0, 633.0], dtype=float)
    fl = [0, 0, 0, 220, 220.1, 240]
    result = ASRResult(
        data=dict(
            wavelength_w=wl,
            freqs_l=fl,
            amplitudes_vvwl=np.ones((3, 3, len(wl), len(fl)),
                                    dtype=complex)),
        metadata={'asr_name': 'asr.ramanpol'})
    write_file('results-asr.ramanpol.json', result.format_as('json'))

    # Check the webpanel
    content = get_webcontent()
    assert 'ramanpol' in content
