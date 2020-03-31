import pytest
from pytest import approx
from .conftest import test_materials
import numpy as np


@pytest.mark.ci
@pytest.mark.parametrize("atoms", test_materials)
def test_borncharges(separate_folder, mockgpaw, mocker, atoms):
    from gpaw import GPAW
    from asr.borncharges import main

    natoms = len(atoms)

    # Number of electrons on each atom
    Z_a = [-2 if ia % 2 else -2 for ia in range(natoms)]
    positive_charge = 2

    # This controls the positive charge of the ions
    def _get_setup_nvalence(self, element_number):
        return positive_charge

    # This controls the electronic contribution to the berry phase
    def _get_berry_phases(self, dir=0, spin=0):
        phase_c = 2 * np.pi * np.dot(Z_a, self.spos_ac) / 2  # / 2 is for spin
        return [phase_c[dir]]

    mocker.patch.object(GPAW, '_get_setup_nvalence', new=_get_setup_nvalence)
    mocker.patch.object(GPAW, '_get_berry_phases', new=_get_berry_phases)

    atoms.write('structure.json')
    results = main()

    Z_analytical_avv = np.array([(Z + positive_charge) * np.eye(3) for Z in Z_a])
    Z_avv = np.array(results['Z_avv'])
    assert Z_analytical_avv == approx(Z_avv)
