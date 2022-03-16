import pytest
from pytest import approx


@pytest.mark.ci
@pytest.mark.parametrize('topology', ['Not checked', 'Z2=1,C_M=1'])
def test_berry(asr_tmpdir_w_params, test_material, mockgpaw, mocker,
               get_webcontent, topology):
    import numpy as np
    from asr.berry import calculate

    test_material.write('structure.json')
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
    results = calculate()

    # check that all phi_km and s_km are returned by asr.berry@calculate
    # note that asr.berry@calculate does not return any phi_km, s_km for 1D materials
    directions = []
    if nd == 3:
        directions = ['0', '1', '2', '0_pi']
    elif nd == 2:
        directions = ['0']
    for d in directions:
        assert results[f'phi{d}_km'] == approx(np.zeros([kpar, nbands]))
        assert results[f's{d}_km'] == approx(np.zeros([kpar, nbands]))

    if topology != 'Not checked':
        # write topology.dat
        from ase.parallel import paropen
        f = paropen('topology.dat', 'w')
        print(topology, file=f, end='')
        f.close()

    from asr.berry import main
    results = main()
    assert results['Topology'] == topology
    get_webcontent()
