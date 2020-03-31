from .conftest import BN, get_webcontent
import pytest


@pytest.mark.ci
@pytest.mark.parametrize("atoms", [BN])
@pytest.mark.parametrize("nspins", [1, 2])
def test_piezoelectrictensor(separate_folder, mockgpaw, mocker, atoms, nspins):
    import numpy as np
    from gpaw import GPAW
    from ase.units import Bohr

    cell_cv = atoms.get_cell() / Bohr
    pbc_c = atoms.get_pbc()
    inv_cell_vc = np.linalg.inv(cell_cv)
    dpde_cvv = np.pi * np.array([np.eye(3) * pbc_c[None] * pbc_c[:, None],
                                 np.zeros((3, 3)), np.zeros((3, 3))])

    def _get_berry_phases(self, dir=0, spin=0):
        strain_vv = np.dot(inv_cell_vc, self.atoms.get_cell() / Bohr) - np.eye(3)
        phase_c = np.dot(dpde_cvv.reshape(3, -1), strain_vv.reshape(-1))
        return [phase_c[dir]]

    def _get_setup_nvalence(self, element_number):
        return 0.0

    def get_number_of_spins(self):
        return nspins

    mocker.patch.object(GPAW, '_get_berry_phases', new=_get_berry_phases)
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
    eps_analytic_vvv = np.tensordot(cell_cv, dpde_cvv,
                                    axes=([0], [0])) / (2 * np.pi * vol) * N * 2
    eps_vvv = results['eps_vvv']
    eps_clamped_vvv = results['eps_clamped_vvv']
    assert eps_vvv == pytest.approx(eps_analytic_vvv)
    assert eps_clamped_vvv == pytest.approx(eps_analytic_vvv)
    assert "Piezoelectrictensor" in content, content
