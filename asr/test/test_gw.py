import pytest


@pytest.mark.ci
def test_gw(asr_tmpdir_w_params, test_material,
            mockgpaw, mocker, get_webcontent, fast_calc):
    import numpy as np
    import gpaw
    from asr.structureinfo import main as structinfo
    from gpaw.response.g0w0 import G0W0
    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    gpaw.GPAW._get_band_gap.return_value = 1
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    gpaw.GPAW._get_fermi_level.return_value = 0.5

    #gs = GS(atoms=test_material, calculator=fast_calc)
    #bs = BS(gs=gs)

    from asr.c2db.gw import gw_main
    ndim = sum(test_material.pbc)

    def calculate(self):
        eps = self.calc.get_all_eigenvalues()[np.newaxis, :, 0:6]

        return {"qp": eps,
                "Z": np.zeros_like(eps) + 1,
                "eps": eps}

    mocker.patch.object(G0W0, "calculate", calculate)
    if ndim > 1:
        results = gw_main(
            atoms=test_material,
            calculator=fast_calc,
            npoints=10,
            kptdensity=2
        )
        assert results['gap_gw'] == pytest.approx(1)
        structinfo(atoms=test_material)
        # test_material.write("structure.json")

        # get_webcontent()
    else:
        with pytest.raises(NotImplementedError):
            gw_main(
                atoms=test_material,
                calculator=fast_calc,
                npoints=10,
                kptdensity=2,
            )
