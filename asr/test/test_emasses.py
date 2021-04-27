import pytest
from pytest import approx
import numpy as np

vbmasses = [0.5, 2.0]
cbmasses = [0.5, 2.0]


def resultstest(results, vbmass, cbmass):
    masses = []
    for k in results:
        if "mass" in k:
            masses.append(results[k])
            if results[k] is None:
                continue
            elif "vb" in k:
                assert -results[k] == approx(vbmass, rel=1e-3), "vb: {}".format(masses)
            else:
                assert results[k] == approx(cbmass, rel=1e-3), masses

        elif "(" in k and ")" in k:
            for k2 in results[k]:
                if "orderMAE" in k2:
                    assert results[k][k2] == approx(0)


@pytest.mark.ci
@pytest.mark.parametrize('gap', [2])
@pytest.mark.parametrize('fermi_level', [0.1])
@pytest.mark.parametrize('vbmass', vbmasses)
@pytest.mark.parametrize('cbmass', cbmasses)
def test_emasses_freelectron(asr_tmpdir_w_params, mockgpaw, mocker,
                             test_material, gap, fermi_level,
                             vbmass, cbmass):
    from asr.emasses import main
    import gpaw

    unpatched = gpaw.GPAW.get_all_eigenvalues

    def get_all_eigs(self):
        res_kn = unpatched(self)
        nvalencebands = self.get_number_of_valence_electrons() // 2
        res_kn[:, :nvalencebands] *= 1 / vbmass
        res_kn[:, nvalencebands:] *= 1 / cbmass
        print('res_kn[0]', res_kn[0])
        return res_kn

    mocker.patch.object(gpaw.GPAW, '_get_band_gap')
    mocker.patch.object(gpaw.GPAW, '_get_fermi_level')
    mocker.patch.object(gpaw.GPAW, 'get_all_eigenvalues', get_all_eigs)
    gpaw.GPAW._get_band_gap.return_value = gap
    gpaw.GPAW._get_fermi_level.return_value = fermi_level

    test_material.write('structure.json')

    results = main()
    resultstest(results, vbmass, cbmass)


@pytest.mark.ci
@pytest.mark.parametrize('gap', [2])
@pytest.mark.parametrize('fermi_level', [0.1])
@pytest.mark.parametrize('vbmass', vbmasses)
@pytest.mark.parametrize('cbmass', cbmasses)
def test_emasses_indirect(asr_tmpdir_w_params, mockgpaw, mocker,
                          test_material, gap, fermi_level,
                          vbmass, cbmass):
    from asr.emasses import main
    import gpaw

    def get_all_eigs(self):
        res_kn = _get_all_eigenvalues(self)
        n = self.get_number_of_valence_electrons() // 2
        res_kn[:, :n] *= 1 / vbmass
        res_kn[:, n:] *= 1 / cbmass
        return res_kn

    mocker.patch.object(gpaw.GPAW, '_get_band_gap')
    mocker.patch.object(gpaw.GPAW, '_get_fermi_level')
    mocker.patch.object(gpaw.GPAW, 'get_all_eigenvalues', get_all_eigs)
    gpaw.GPAW._get_band_gap.return_value = gap
    gpaw.GPAW._get_fermi_level.return_value = fermi_level

    test_material.write('structure.json')

    results = main()
    resultstest(results, vbmass, cbmass)

    # check location of minmax


def _get_all_eigenvalues(self):
    from ase.units import Bohr, Ha
    icell = self.atoms.cell.reciprocal() * 2 * np.pi * Bohr
    n = self.parameters.gridsize

    offsets = np.indices((n, n, n)).T.reshape((n ** 3, 1, 3)) - n // 2

    maxk = np.max(self.kpts)
    dim = self.atoms.pbc.sum()
    if dim == 1:
        kvec = np.array([0, 0, maxk * 0.4])
    else:
        kvec = np.array([maxk * 0.4, 0, 0])
    ceps_kn = 0.5 * (np.dot((self.kpts - kvec) + offsets, icell) ** 2).sum(2).T
    ceps_kn.sort()

    veps_kn = 0.5 * (np.dot(self.kpts + offsets, icell) ** 2).sum(2).T
    veps_kn.sort()

    nvalencebands = self.get_number_of_valence_electrons() // 2
    gap = self._get_band_gap()
    eps_kn = np.concatenate((-veps_kn[:, ::-1][:, -nvalencebands:],
                            ceps_kn + gap / Ha), axis=-1)

    nbands = self.get_number_of_bands()
    return eps_kn[:, :nbands] * Ha


# @pytest.mark.parametrize('gap', [2])
# @pytest.mark.parametrize('fermi_level', [0.1])
# @pytest.mark.parametrize('vbmass', vbmasses)
# @pytest.mark.parametrize('cbmass', cbmasses)
# def test_emasses_rashba(asr_tmpdir_w_params, mockgpaw, mocker,
#                              test_material, gap, fermi_level,
#                              vbmass, cbmass):
#     from asr.emasses import main
#     import gpaw


#     unpatched = gpaw.GPAW.get_all_eigenvalues
#     def get_all_eigs(self):
#         res_kn = _get_all_eigenvalues_rashba(self)
#         n = self.get_number_of_electrons()
#         res_kn[:, :n] *= 1 / vbmass
#         res_kn[:, n:] *= 1 / cbmass
#         return res_kn


#     mocker.patch.object(gpaw.GPAW, '_get_band_gap')
#     mocker.patch.object(gpaw.GPAW, '_get_fermi_level')
#     mocker.patch.object(gpaw.GPAW, 'get_all_eigenvalues', get_all_eigs)
#     gpaw.GPAW._get_band_gap.return_value = gap
#     gpaw.GPAW._get_fermi_level.return_value = fermi_level

#     test_material.write('structure.json')

#     results = main()
#     try:
#         resultstest(results, vbmass, cbmass)
#     except Exception as e:
#         import matplotlib.pyplot as plt

#         calc = gpaw.GPAW("_refined.gpw")
#         print(calc.kpts.shape)
#         print(gpaw.GPAW("gs.gpw").kpts.shape)
#         eps_kn = calc.get_all_eigenvalues()
#         nelec = calc.get_number_of_electrons()
#         plt.plot(eps_kn[:, nelec + 1])
#         plt.show()
#         raise e


# def _get_all_eigenvalues_rashba(self):
#     from ase.units import Bohr, Ha
#     icell = self.atoms.cell.reciprocal() * 2 * np.pi * Bohr
#     n = self.parameters.gridsize


#     offsets = np.indices((n, n, n)).T.reshape((n ** 3, 1, 3)) - n // 2
#     maxk = np.max(self.kpts)
#     dim = self.atoms.pbc.sum()

#     perc = 0.01
#     if dim == 1:
#         kvec = np.array([0, 0, maxk * perc])
#     else:
#         kvec = np.array([maxk * perc, 0, 0])
#     # ceps_kn = 0.5 * (np.dot(self.kpts  + offsets, icell) ** 2).sum(2).T
#     # ceps_kn.sort()

#     ceps1_kn = 0.5 *
# (np.dot(self.kpts - kvec  + offsets, icell) ** 2).sum(2).T
#     ceps2_kn = 0.5 *
# (np.dot(self.kpts + kvec  + offsets, icell) ** 2).sum(2).T + 0.0 * Ha
#     ceps_kn = np.concatenate((ceps1_kn, ceps2_kn), axis=-1)
#     ceps_kn.sort()
#     # delta = (np.min(ceps_kn[:, 0]) - np.min(ceps_kn[:, 1]))

#     veps_kn = 0.5 * (np.dot(self.kpts + offsets, icell) ** 2).sum(2).T
#     veps_kn.sort()

#     # ceps_kn[:, 1] += (0.5 *
# (np.dot((self.kpts - kvec)  + offsets, icell) ** 2).sum(2).T)[:, 1]
#     # ceps_kn[:, 1] += -(0.5 *
# (np.dot(self.kpts  + offsets, icell) ** 2).sum(2).T)[:, 1]
#     # delta = (np.min(ceps_kn[:, 0]) - np.min(ceps_kn[:, 1]))
#     # ceps_kn[:, 1] += delta
#     # ceps_kn.sort()

#     nelectrons = self.get_number_of_electrons()
#     gap = self._get_band_gap()
#     eps_kn = np.concatenate((-veps_kn[:, ::-1][:, -nelectrons:],
#                             ceps_kn + gap / Ha), axis=-1)


#     nbands = self.get_number_of_bands()
#     return eps_kn[:, :nbands] * Ha
