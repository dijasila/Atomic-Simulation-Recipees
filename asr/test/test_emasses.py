import pytest
from pytest import approx
import numpy as np


@pytest.mark.parametrize('gap', [1])
@pytest.mark.parametrize('fermi_level', [0.1])
@pytest.mark.parametrize('vbmass', [0.5, 1, 2.0])
@pytest.mark.parametrize('cbmass', [0.5, 1, 2.0])
def test_emasses(asr_tmpdir_w_params, mockgpaw, mocker,
                 test_material, gap, fermi_level,
                 vbmass, cbmass):
    from asr.emasses import main
    import gpaw

    def get_all_eigs(self):
        res_kn = _get_all_eigenvalues(self)
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

    
def _get_all_eigenvalues(self):
    from ase.units import Bohr, Ha
    icell = self.atoms.get_reciprocal_cell() * 2 * np.pi * Bohr
    n = self.parameters.gridsize
    offsets = np.indices((n, n, n)).T.reshape((n ** 3, 1, 3)) - n // 2
    eps_kn = 0.5 * (np.dot(self.kpts + offsets, icell) ** 2).sum(2).T
    eps_kn.sort()
    
    nelectrons = self.get_number_of_electrons()
    gap = self._get_band_gap()
    eps_kn = np.concatenate(
        (-eps_kn[:, ::-1][:, -nelectrons:],
         eps_kn + gap / Ha),
        axis=1,
    )
    nbands = self.get_number_of_bands()
    return eps_kn[:, :nbands] * Ha
