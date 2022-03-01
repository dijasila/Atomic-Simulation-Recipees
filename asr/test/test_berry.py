import pytest
from pytest import approx


@pytest.mark.ci
@pytest.mark.parametrize('topology', ['Not checked', 'Z2=1,C_M=1'])
def test_berry(test_material, mockgpaw, mocker,
               get_webcontent, fast_calc, topology):
    import numpy as np
    from asr.c2db.berry import calculate

    kpar = 10
    nbands = 2

    nd = np.sum(test_material.pbc)
    if nd < 2:
        pytest.xfail("Expected fail: berry not implemented for <2D.")

    def parallel_transport(calc='gs_berry.gpw', direction=0, theta=0, phi=0):
        phi_km = np.zeros([kpar, nbands])
        s_km = np.zeros([kpar, nbands])
        return phi_km, s_km

    mocker.patch('gpaw.berryphase.parallel_transport', create=True,
                 new=parallel_transport)
    results = calculate(atoms=test_material,
                        calculator=fast_calc)

    # check that all phi_km and s_km are returned by asr.c2db.berry@calculate
    # note that asr.c2db.berry@calculate does not return
    # any phi_km, s_km for 1D materials
    directions = []
    if nd == 3:
        directions = ['0', '1', '2', '0_pi']
    elif nd == 2:
        directions = ['0']
    for d in directions:
        assert results[f'phi{d}_km'] == approx(np.zeros([kpar, nbands]))
        assert results[f's{d}_km'] == approx(np.zeros([kpar, nbands]))

    from asr.c2db.berry import main
    results = main(
        atoms=test_material,
        topology=topology,
        calculator=fast_calc,
    )
    assert results['Topology'] == topology
    test_material.write('structure.json')
    get_webcontent()
