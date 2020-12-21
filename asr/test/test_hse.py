import pytest


@pytest.mark.ci
def test_hse(asr_tmpdir_w_params, test_material, mockgpaw, mocker,
             get_webcontent):
    import gpaw
    from pathlib import Path
    import numpy as np
    from asr.hse import main
    from asr.structureinfo import main as structinfo

    test_material.write('structure.json')

    def non_self_consistent_eigenvalues(calc,
                                        xcname,
                                        n1,
                                        n2,
                                        kpt_indices=None,
                                        snapshot=None,
                                        ftol=42.0):
        Path(snapshot).write_text('{}')
        e_skn = calc.eigenvalues[np.newaxis, :, n1:n2].copy()
        v_skn = np.zeros_like(e_skn)
        v_hyb_skn = np.zeros_like(e_skn)
        v_hyb_skn[:, :, :2] = 0.0
        v_hyb_skn[:, :, 2:] = 1.0
        return e_skn, v_skn, v_hyb_skn

    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    gpaw.GPAW._get_fermi_level.return_value = 0.5
    gpaw.GPAW._get_band_gap.return_value = 1

    mocker.patch('gpaw.hybrids.eigenvalues.non_self_consistent_eigenvalues',
                 create=True, new=non_self_consistent_eigenvalues)

    def vxc(calc, xc):
        return calc.eigenvalues[np.newaxis]

    mocker.patch('gpaw.xc.tools.vxc', create=True, new=vxc)
    results = main(atoms=test_material).result
    assert results['gap_hse_nosoc'] == pytest.approx(2.0)
    assert results['gap_dir_hse_nosoc'] == pytest.approx(2.0)

    # We need to call structureinfo in order to make the webpanel.
    # This should be fixed in the future.
    structinfo(atoms=test_material)
    html = get_webcontent()
    assert 'hse-bs.png' in html
