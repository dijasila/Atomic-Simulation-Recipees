import pytest


@pytest.mark.ci
def test_raman(asr_tmpdir_w_params, asr_tmpdir, test_material, get_webcontent):
    from asr.core import write_file, ASRResult
    import numpy as np

    test_material.write('structure.json')

    wl = np.array([488.0, 532.0, 633.0], dtype=float)
    fl = [0, 0, 0, 220, 220.1, 240]
    result = ASRResult(
        data=dict(
            wavelength_w=wl,
            freqs_l=fl,
            amplitudes_vvwl=np.ones((3, 3, len(wl), len(fl)),
                                    dtype=complex)),
        metadata={'asr_name': 'asr.raman'})
    write_file('results-asr.raman.json', result.format_as('json'))

    # Check the webpanel
    content = get_webcontent()
    assert 'raman' in content
