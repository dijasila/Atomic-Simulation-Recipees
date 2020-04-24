import pytest
from pytest import approx
import numpy as np


@pytest.mark.parametrize('gap', [2])
@pytest.mark.parametrize('fermi_level', [0.1])
@pytest.mark.parametrize('vbmass', [0.01, 0.5, 2.0, 20])
@pytest.mark.parametrize('cbmass', [0.01, 0.5, 2.0, 20])
def test_emasses(asr_tmpdir_w_params, mockgpaw, mocker,
                 test_material, gap, fermi_level,
                 vbmass, cbmass):
    from asr.emasses import main
    import gpaw

    
    unpatched = gpaw.GPAW.get_all_eigenvalues
    def get_all_eigs(self):
        res_kn = unpatched(self)
        res_kn[:, :self.get_number_of_electrons()] *= 1 / vbmass
        res_kn[:, self.get_number_of_electrons():] *= 1 / cbmass
        return res_kn

    mocker.patch.object(gpaw.GPAW, '_get_band_gap')
    mocker.patch.object(gpaw.GPAW, '_get_fermi_level')
    mocker.patch.object(gpaw.GPAW, 'get_all_eigenvalues', get_all_eigs)
    gpaw.GPAW._get_band_gap.return_value = gap
    gpaw.GPAW._get_fermi_level.return_value = fermi_level
    

    test_material.write('structure.json')

    results = main()

    for k in results:
        if "mass" in k:
            if results[k] is None:
                continue
            elif "vb" in k:
                assert -results[k] == approx(vbmass)
            else:
                assert results[k] == approx(cbmass)

        elif "(" in k and ")" in k:
            for k2 in results[k]:
                if "orderMAE" in k2:
                    assert results[k][k2] == approx(0)

    
