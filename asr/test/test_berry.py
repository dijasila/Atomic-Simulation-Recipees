import pytest
from pytest import approx


@pytest.mark.ci
def test_berry(asr_tmpdir_w_params, test_material, mockgpaw, mocker, get_webcontent):
    import numpy as np
    test_material.write('structure.json')
    kpar = 10
    nbands = 2

    def parallel_transport(calc='gs_berry.gpw', direction=0, theta=0, phi=0):
        phi_km = np.zeros([kpar, nbands])
        s_km = np.zeros([kpar, nbands])
        return phi_km, s_km

    from asr.berry import calculate, plot_phases
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
        plot_phases(name=d)

    from asr.berry import main
    results = main()
    assert results['Topology'] == 'Not checked'
    get_webcontent()


@pytest.mark.ci
@pytest.mark.parametrize('topology', ['Z2=1,C_M=1'])
def test_berry_nontrivial(asr_tmpdir_w_params, mockgpaw, mocker, topology, get_webcontent):
    from ase.build.surface import graphene
    structure = graphene()
    structure.write('structure.json')

    # write phi0_km, s0_km
    from asr.core import write_json
    import numpy as np
    dct = {}
    dct['phi0_km'] = np.zeros([10, 2])
    dct['s0_km'] = np.zeros([10, 2])
    write_json('results-asr.berry@calculate.json', dct)

    # write topology.dat
    from pathlib import Path
    from ase.parallel import paropen
    f = paropen('topology.dat', 'w')
    print(topology, file=f)
    f.close()

    from asr.berry import main
    results = main()
    assert results['Topology'] == topology + '\n'
    content = get_webcontent()
    assert f"<td>Bandtopology</td><td>{topology}</td>" in \
        content, content
