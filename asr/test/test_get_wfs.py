import pytest
import numpy as np


# @pytest.mark.ci
# def test_get_wfs(asr_tmpdir, test_material):
#     from pathlib import Path
#     from ase.io import write
#     from asr.gs import calculate, main
#     from asr.get_wfs import main as get_wfs
# 
#     write('structure.json', test_material)
#     calculate(
#         calculator={
#             "name": "gpaw",
#             "kpts": {"density": 2, "gamma": True},
#         },
#     )
# 
#     main()
#     results = get_wfs()
# 
#     assert Path('wf.0_0.cube').is_file()
#     assert results['above_below'] == (None, None)

@pytest.mark.parametrize('ef', np.arange(-5, 3, 0.5))
@pytest.mark.ci
def test_return_erange_states(ef):
    from asr.get_wfs import return_erange_states
    import numpy as np

    erange = (-0.8, 0.8)
    evs = np.arange(-5, 3, 0.1)
    ref_evs = evs - ef

    states = return_erange_states(evs, ef, erange)
    ref_states = np.where((erange[1] > ref_evs)
                          & (ref_evs > erange[0]))[0]

    assert all(states == ref_states)
