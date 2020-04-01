from .conftest import test_materials, get_webcontent
from ase.units import Bohr
import numpy as np
import pytest


def zero_pad_non_pbc_strain_directions(matrix_cvv, pbc_c):
    return matrix_cvv * pbc_c[None] * pbc_c[None, None]


def get_strain_from_atoms(inv_cell_vc, atoms):
    return np.dot(inv_cell_vc, atoms.get_cell() / Bohr) - np.eye(3)


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
@pytest.mark.parametrize("nspins", [1, 2])
def test_piezoelectrictensor(separate_folder, mockgpaw, mocker, atoms, nspins):
    import numpy as np
    from gpaw import GPAW

    cell_cv = atoms.get_cell() / Bohr
    pbc_c = atoms.get_pbc()
    inv_cell_vc = np.linalg.inv(cell_cv)
    # dphase_c / dstrain_vv
    dpde_cvv = zero_pad_non_pbc_strain_directions(
        np.array([np.eye(3)] * 3),
        pbc_c,
    )

    # ddipol_v / dstrain_vv
    ddipolde_vvv = zero_pad_non_pbc_strain_directions(
        np.array([np.eye(3)] * 3),
        pbc_c,
    )

    def _get_berry_phases(self, dir=0, spin=0):
        strain_vv = get_strain_from_atoms(inv_cell_vc, self.atoms)
        phase_c = np.dot(dpde_cvv.reshape(3, -1), strain_vv.reshape(-1))
        return [phase_c[dir]]

    def _get_dipole_moment(self):
        strain_vv = get_strain_from_atoms(inv_cell_vc, self.atoms)
        dipol_v = np.dot(ddipolde_vvv.reshape(3, -1), strain_vv.reshape(-1))
        return dipol_v

    def _get_setup_nvalence(self, element_number):
        return 1

    def get_number_of_spins(self):
        return nspins

    mocker.patch.object(GPAW, '_get_berry_phases', new=_get_berry_phases)
    mocker.patch.object(GPAW, '_get_dipole_moment', new=_get_dipole_moment)
    mocker.patch.object(GPAW, '_get_setup_nvalence', new=_get_setup_nvalence)
    mocker.patch.object(GPAW, 'get_number_of_spins', new=get_number_of_spins)
    from ase.io import write
    from asr.piezoelectrictensor import main
    write('structure.json', atoms)
    results = main()
    content = get_webcontent('database.db')

    N = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
    vol = atoms.get_volume() / Bohr**3

    # Formula for piezoeletric tensor
    # (The last factor of 2 is for spins)
    eps_berry_analytic_vvv = np.tensordot(
        cell_cv, dpde_cvv,
        axes=([0], [0])
    ) / (2 * np.pi * vol) * N * 2

    eps_dipole_analytic_vvv = ddipolde_vvv / (vol * Bohr) * N

    eps_analytic_vvv = np.zeros((3, 3, 3), float)
    eps_analytic_vvv[pbc_c] = eps_berry_analytic_vvv[pbc_c]
    eps_analytic_vvv[~pbc_c] = eps_dipole_analytic_vvv[~pbc_c]

    eps_vvv = results['eps_vvv']
    eps_clamped_vvv = results['eps_clamped_vvv']
    assert eps_vvv == pytest.approx(eps_analytic_vvv)
    assert eps_clamped_vvv == pytest.approx(eps_analytic_vvv)
    assert "Piezoelectrictensor" in content, content
