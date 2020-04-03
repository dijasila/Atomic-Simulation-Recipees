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

    atoms.write('structure.json')
    results = main()

    Z_analytical_avv = np.array([(Z + positive_charge) * np.eye(3) for Z in Z_a])
    Z_avv = np.array(results['Z_avv'])
    assert Z_analytical_avv == approx(Z_avv)


@pytest.mark.integration_test_gpaw
def test_gpaw_berry_get_berry_phases_integration(separate_folder):
    from .conftest import BN
    from asr.borncharges import main
    from asr.setup.params import main as setupparams

    calculator_params = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 300},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 2.0, 'gamma': True},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'convergence': {'bands': 'CBM+3.0'},
        'nbands': '100%',
        'txt': 'gs.txt',
        'charge': 0,
    }

    formal_calculator = {
        'name': 'gpaw',
        'mode': {'name': 'pw', 'ecut': 300},
        'xc': 'PBE',
        'basis': 'dzp',
        'kpts': {'density': 2.0},
        'occupations': {'name': 'fermi-dirac',
                        'width': 0.05},
        'txt': 'formalpol.txt',
        'charge': 0
    }
    setupparams(params={'asr.gs@calculate': {'calculator': calculator_params},
                        'asr.formalpolarization': {'calculator': formal_calculator}})
    BN.write('structure.json')

    results = main()

    ZB_vv = np.eye(3)
    diag = np.eye(3, dtype=bool)
    ZB_vv[diag] = [2.71, 2.71, 0.27]
    for Z_vv, sym in zip(results['Z_avv'], results['sym_a']):
        if sym == 'B':
            sign = 1
        else:
            sign = -1

        assert np.all(sign * Z_vv[diag] > 0)
        assert Z_vv[~diag] == pytest.approx(0, abs=0.001)
        assert Z_vv == pytest.approx(sign * ZB_vv, abs=0.5)
