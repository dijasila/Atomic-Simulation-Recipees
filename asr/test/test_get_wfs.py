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


@pytest.mark.parametrize('setup_method', ['uni', 'gen'])
@pytest.mark.ci
def test_return_defect_index(asr_tmpdir, setup_method):
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
    if setup == 'uni':
        setup()
    else:
        setup(general_algorithm=15.)
    p = Path('.')
    pathlist = list(p.glob('defects.*/charge_0/'))
    for path in pathlist:
        defname = str(path.absolute()).split('/')[-2].split('.')[-1]
        structure = read(path / 'unrelaxed.json')
        def_index, is_vacancy = return_defect_index(
            path, primitive, structure)

        assert results[defname][0] == def_index
        assert results[defname][1] == is_vacancy

    pristine = primitive.repeat((3, 3, 1))
    try:
        def_index, is_vacancy = return_defect_index(
            path, primitive, pristine)
    except AssertionError:
        # function should fail with an assertion error
        # when the input is not a defect structure but
        # a pristine one
        assert True


@pytest.mark.parametrize('gap', np.arange(0, 2.01, 20))
@pytest.mark.ci
def test_get_above_below(gap):
    from asr.get_wfs import get_above_below

    evs = np.arange(-5, 5.5, 20)
    ef = 0
    vbm = ef - gap / 2.
    cbm = ef + gap / 2.

    above_below = get_above_below(evs, ef, vbm, cbm)
    if gap >= 1:
        ref = (True, True)
    else:
        ref = (False, False)

    assert above_below[0] == ref[0]
    assert above_below[1] == ref[1]


@pytest.mark.parametrize('formula', ['MoS2', 'MoSe2', 'MoTe2'])
@pytest.mark.ci
def test_get_reference_index(formula):
    from asr.get_wfs import get_reference_index
    from ase.build import mx2

    atoms = mx2(formula)
    atoms = atoms.repeat((3, 3, 1))
    ref_index = get_reference_index(0, atoms)

    assert (ref_index == 15 or ref_index == 10)
