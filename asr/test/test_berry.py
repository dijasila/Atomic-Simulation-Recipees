import pytest 
from pytest import approx


def test_berry(asr_tmpdir_w_params, test_material, mockgpaw, mocker, get_webcontent):
    import numpy as np
    test_material.write('structure.json')
    kpar = 10
    nbands = 2
    
    def parallel_transport(calc='gs_berry.gpw', direction=0, theta=0, phi=0):
        phi_km = np.zeros([kpar, nbands])
        s_km = np.zeros([kpar, nbands])
        return phi_km, s_km

    from asr.berry import calculate, main
    mocker.patch('gpaw.berryphase.parallel_transport', create=True,
                 new=parallel_transport)
    results = calculate()

    # check that all phi_km and s_km are returned by asr.berry@calculate 
    # note that asr.berry@calculate does not return any phi_km, s_km for 1D materials
    nd = np.sum(test_material.pbc)
    directions = []
    if nd == 3:
        directions = ['0', '1', '2', '0_pi']
    elif nd == 2:
        directions = ['0']       
    for d in directions:
        assert results[f'phi{d}_km'] == approx(np.zeros([kpar, nbands]))
        assert results[f's{d}_km'] == approx(np.zeros([kpar, nbands]))

    main()
    get_webcontent()
