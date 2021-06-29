import pytest
from pytest import approx
import numpy as np


@pytest.mark.ci
def test_borncharges(
        asr_tmpdir_w_params, mockgpaw, mocker, test_material, fast_calc):
    from gpaw import GPAW
    from asr.borncharges import main

    natoms = len(test_material)

    # Number of electrons on each atom
    Z_a = np.array([-2 if ia % 2 else -2 for ia in range(natoms)])
    positive_charge = 1

    # This controls the positive charge of the ions
    def _get_setup_nvalence(self, element_number):
        return positive_charge

    # The dipole moment is used for non-periodic directions
    def _get_dipole_moment(self):
        return np.dot(Z_a + positive_charge, self.atoms.get_positions())

    # This controls the electronic contribution to the berry phase
    def _get_berry_phases(self, dir=0, spin=0):
        phase_c = 2 * np.pi * np.dot(Z_a, self.spos_ac) / 2  # / 2 is for spin
        return [phase_c[dir]]

    mocker.patch.object(GPAW, '_get_setup_nvalence', new=_get_setup_nvalence)
    mocker.patch.object(GPAW, '_get_dipole_moment', new=_get_dipole_moment)
    mocker.patch.object(GPAW, '_get_berry_phases', new=_get_berry_phases)

    results = main(
        atoms=test_material,
        calculator=fast_calc,
    )

    Z_analytical_avv = np.array([
        (Z + positive_charge) * np.eye(3) for Z in Z_a])
    Z_avv = np.array(results['Z_avv'])
    assert Z_analytical_avv == approx(Z_avv)
