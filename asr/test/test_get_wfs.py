import pytest
import numpy as np


@pytest.mark.parametrize('ef', np.arange(-5, 3, 2))
@pytest.mark.parametrize('emin', [-0.8, -0.2, 1])
@pytest.mark.parametrize('emax', [0.13, 0.5, -0.1])
@pytest.mark.ci
def test_return_erange_states(ef, emin, emax):
    from asr.get_wfs import return_erange_states
    import numpy as np

    erange = (emin, emax)
    evs = np.arange(-5, 3, 0.1)
    ref_evs = evs - ef

    states = return_erange_states(evs, ef, erange)
    ref_states = np.where((erange[1] > ref_evs)
                          & (ref_evs > erange[0]))[0]

    assert all(states == ref_states)

    # for inverted energy range there should be no
    # states returned
    if emin > emax:
        assert len(states) == 0


@pytest.mark.ci
def test_return_defect_index(asr_tmpdir):
    from pathlib import Path
    from .materials import BN
    from ase.io import read, write
    from asr.setup.defects import main as setup
    from asr.get_wfs import return_defect_index

    results = {'v_B': (0, True),
               'v_N': (1, True),
               'N_B': (0, False),
               'B_N': (1, False)}

    primitive = BN.copy()
    write('unrelaxed.json', primitive)
    setup()
    p = Path('.')
    pathlist = list(p.glob('defects.*/charge_0/'))
    for path in pathlist:
        defname = str(path.absolute()).split('/')[-2].split('.')[-1]
        structure = read(path / 'unrelaxed.json')
        def_index, is_vacancy = return_defect_index(
            path, primitive, structure)

        assert results[defname][0] == def_index
        assert results[defname][1] == is_vacancy
