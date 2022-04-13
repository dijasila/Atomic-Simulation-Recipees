import pytest

from contextlib import nullcontext
from asr.c2db.gs import GSWorkflow
from asr.c2db.bandstructure import BSWorkflow
from asr.c2db.gw import GWWorkflow


def gw_workflow(rn, atoms, calculator):
    gsw = GSWorkflow(rn, atoms=atoms, calculator=calculator)
    bsw = BSWorkflow(rn, gsworkflow=gsw, npoints=10)
    return GWWorkflow(rn, bsworkflow=bsw, kptdensity=2)



@pytest.mark.ci
def test_gw(repo, asr_tmpdir_w_params, test_material,
            mockgpaw, mocker, get_webcontent, fast_calc):
    import numpy as np
    import gpaw
    from asr.structureinfo import main as structinfo
    from gpaw.response.g0w0 import G0W0
    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    gpaw.GPAW._get_band_gap.return_value = 1
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    gpaw.GPAW._get_fermi_level.return_value = 0.5

    ndim = sum(test_material.pbc)

    def calculate(self):
        eps = self.calc.get_all_eigenvalues()[np.newaxis, :, 0:6]

        return {"qp": eps,
                "Z": np.zeros_like(eps) + 1,
                "eps": eps}

    mocker.patch.object(G0W0, "calculate", calculate)

    with repo:
        gwworkflow = repo.run_workflow(gw_workflow, atoms=test_material,
                                       calculator=fast_calc)

    if ndim > 1:
        expectation = nullcontext()
    else:
        expectation = pytest.raises(NotImplementedError)

    with expectation:
        gwworkflow.postprocess.runall_blocking(repo)

    if ndim > 1:
        with repo:
            results = gwworkflow.postprocess.value().output
        assert results['gap_gw'] == pytest.approx(1)
        structinfo(atoms=test_material)
        # get_webcontent()
