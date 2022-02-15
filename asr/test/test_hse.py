import pytest


@pytest.fixture
def gsresult(mockgpaw, asr_tmpdir_w_params, test_material, fast_calc):
    from asr.c2db.gs import calculate
    gsresult = calculate(atoms=test_material, calculator=fast_calc)
    return gsresult


@pytest.fixture
def magstate(gsresult):
    from asr.c2db.magstate import main
    magstate = main(groundstate=gsresult)
    return magstate


@pytest.fixture
def mag_ani(gsresult, magstate):
    from asr.c2db.magnetic_anisotropy import main
    mag_ani = main(groundstate=gsresult,
                   magnetic=magstate['is_magnetic'])
    return mag_ani


@pytest.fixture
def gspostprocess(gsresult, mag_ani):
    from asr.c2db.gs import postprocess
    return postprocess(groundstate=gsresult, mag_ani=mag_ani)


@pytest.fixture
def bsresult(gsresult):
    from asr.c2db.bandstructure import calculate
    return calculate(gsresult=gsresult,
                     npoints=10)


@pytest.fixture
def bspostprocess(bsresult, gsresult, mag_ani, gspostprocess):
    from asr.c2db.bandstructure import postprocess
    return postprocess(bsresult=bsresult, gsresult=gsresult,
                       mag_ani=mag_ani,
                       gspostprocess=gspostprocess)


@pytest.mark.ci
@pytest.mark.parametrize("bandgap", [0, 1])
def test_hse(gsresult, asr_tmpdir_w_params, test_material, mockgpaw, mocker,
             get_webcontent, fast_calc, bandgap, bspostprocess, mag_ani,
             bsresult):
    import gpaw
    from pathlib import Path
    import numpy as np
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
        v_hyb_skn[:, :, 2:] = bandgap
        return e_skn, v_skn, v_hyb_skn

    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    gpaw.GPAW._get_fermi_level.return_value = 0.5
    gpaw.GPAW._get_band_gap.return_value = bandgap

    mocker.patch('gpaw.hybrids.eigenvalues.non_self_consistent_eigenvalues',
                 create=True, new=non_self_consistent_eigenvalues)

    def vxc(calc, xc):
        return calc.eigenvalues[np.newaxis]

    mocker.patch('gpaw.xc.tools.vxc', create=True, new=vxc)

    from asr.c2db.hse import calculate, postprocess

    hse_calculate_result = calculate(
        gsresult=gsresult,
        mag_ani=mag_ani)

    results = postprocess(
        results_hse=hse_calculate_result,
        results_bs_post=bspostprocess,
        results_bs_calculate=bsresult,
        mag_ani=mag_ani)
    assert results['gap_hse_nosoc'] == pytest.approx(2 * bandgap)
    assert results['gap_dir_hse_nosoc'] == pytest.approx(2 * bandgap)
    return

    # We need to call structureinfo in order to make the webpanel.
    # This should be fixed in the future.
    structinfo(atoms=test_material)

    #html = get_webcontent()
    #assert 'hse-bs.png' in html
