import pytest

from asr.c2db.hse import HSEWorkflow
from asr.c2db.gs import GSWorkflow
from asr.c2db.bandstructure import BSWorkflow


def gs_bs_hse_workflow(rn, atoms, calculator):
    gsw = GSWorkflow(rn, atoms=atoms, calculator=calculator)
    bsw = BSWorkflow(rn, gsworkflow=gsw, npoints=10)
    return HSEWorkflow(rn, bsworkflow=bsw)


@pytest.mark.ci
@pytest.mark.parametrize("bandgap", [0, 1])
def test_hse(repo, asr_tmpdir_w_params, test_material, mockgpaw, mocker,
             get_webcontent, fast_calc, bandgap):
    import gpaw
    from pathlib import Path
    import numpy as np

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

    with repo:
        hseworkflow = repo.run_workflow(gs_bs_hse_workflow,
                                        atoms=test_material,
                                        calculator=fast_calc)

    hseworkflow.postprocess.runall_blocking(repo)

    with repo:
        results = hseworkflow.postprocess.value().output
        assert results['gap_hse_nosoc'] == pytest.approx(2 * bandgap)
        assert results['gap_dir_hse_nosoc'] == pytest.approx(2 * bandgap)

    # We need to call structureinfo in order to make the webpanel.
    # This should be fixed in the future.
    # structinfo(atoms=test_material)

    # html = get_webcontent()
    # assert 'hse-bs.png' in html
