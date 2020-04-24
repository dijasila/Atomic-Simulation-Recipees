import pytest


@pytest.mark.ci
def test_bse(asr_tmpdir_w_params, test_material, mockgpaw, mocker, get_webcontent):
    import numpy as np
    import gpaw
    from gpaw.response.bse import BSE
    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    gpaw.GPAW._get_band_gap.return_value = 1.0
    mocker.patch.object(gpaw.GPAW, "get_fermi_level")
    gpaw.GPAW.get_fermi_level.return_value = 0.5

    from asr.bse import main
    test_material.write("structure.json")
    ndim = sum(test_material.pbc)

    def calculate(self):
        E_B = 1.0

        return {"E_B": E_B}

    mocker.patch.object(BSE, "calculate", calculate)
    if ndim > 1:
        main()
        get_webcontent()
    else:
        with pytest.raises(NotImplementedError):
            main()
