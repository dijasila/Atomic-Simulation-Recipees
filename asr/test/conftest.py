import pytest
import os
import numpy as np
from ase.calculators.calculator import kpts2ndarray
from ase.units import Bohr, Ha

from pathlib import Path


class GPAWMock():
    """Mock of GPAW calculator.

    Sets up a free electron like gpaw calculator.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'magmom', 'magmoms']

    default_parameters = {'kpts': np.zeros((1, 3)),
                          'nvalence': 0.0,
                          'nbands': 20,
                          'gridsize': 7}

    class Occupations:

        def todict(self):
            return {'name': 'SomeOccupationName', 'width': 0.05}

    occupations = Occupations()

    class Setups(list):
        nvalence = 20

    from types import SimpleNamespace
    setups = Setups()
    setups.append(SimpleNamespace(symbol='MySymbol', fingerprint='asdf1234'))

    def __init__(self, *args, **kwargs):
        from ase.io import read
        self.atoms = read('structure.json')
        n = 4
        self.nvalence = 4
        self.kpts = kpts2ndarray([n, n, n], self.atoms)
        self.nk = len(self.kpts)
        icell = self.atoms.get_reciprocal_cell() * 2 * np.pi * Bohr
        offsets = np.indices((n, n, n)).T.reshape((n**3, 1, 3)) - n // 2
        eps = 0.5 * (np.dot(self.kpts + offsets, icell)**2).sum(2).T
        eps.sort()
        nbands = 10
        self.eigenvalues = eps[:, :nbands] * Ha
        self.results = {'energy': 0.0,
                        'forces': np.zeros((len(self.atoms), 3), float),
                        'stress': np.zeros((3, 3), float)}

    def get_eigenvalues(self, kpt, spin=0):
        assert spin == 0
        return self.eigenvalues[kpt].copy()

    def get_k_point_weights(self):
        return [1 / self.nk] * self.nk
    
    def get_fermi_level(self):
        v = self.atoms.get_volume() / Bohr**3
        kF = (self.nvalence / v * 3 * np.pi**2)**(1 / 3)
        return 0.5 * kF**2 * Ha

    def get_ibz_k_points(self):
        return self.kpts.copy()

    def get_number_of_spins(self):
        return 1

    def get_number_of_bands(self):
        return 10

    def get_number_of_electrons(self):
        return 4

    def write(self, name):
        Path(name).write_text('Test calculation')

    def read(self, name):
        pass

    def get_electrostatic_potential():
        pass

    def get_property(self, name, atoms=None, allow_calculation=True):
        assert name in self.implemented_properties

        if name == 'magmom':
            return 0.0

        if name == 'energy':
            return self.get_potential_energy(atoms)

        if name == 'dipole':
            return self.get_dipole_moment(atoms)

        if name == 'stress':
            return self.get_stress(atoms)

        if name == 'forces':
            return self.get_forces(atoms)

    def todict(self):
        return {'convergence': {'bands': 4}}

    def get_forces(self, atoms=None):
        if atoms:
            na = len(atoms)
        else:
            na = len(self.atoms)
        return np.zeros((na, 3), float)

    def get_stress(self, atoms=None):
        return np.eye(3)

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return 1234.4321


@pytest.fixture
def mock_GPAW(monkeypatch):
    import gpaw

    def get_spinorbit_eigenvalues(calc, bands=None, gw_kn=None,
                                  return_spin=False,
                                  return_wfs=False, scale=1.0,
                                  theta=0.0, phi=0.0):

        nk = len(calc.get_ibz_k_points())
        nspins = calc.get_number_of_spins()
        nbands = calc.get_number_of_bands()
        bands = list(range(nbands))

        e_ksn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)[bands]
                           for s in range(nspins)] for k in range(nk)])

        s_kvm = np.zeros((nk, 3, nbands), float)
        s_kvm[:, 2, ::2] = 1
        s_kvm[:, 2, ::2] = -1
        e_km = e_ksn.reshape((nk, -1))
        if return_spin:
            return e_km.T, s_kvm
        else:
            return e_km.T

    def occupation_numbers(occ, eps_skn, weight_k, nelectrons):
        return 0, 1, 2, 3

    monkeypatch.setattr(gpaw, 'GPAW', GPAWMock)
    monkeypatch.setattr(gpaw.spinorbit, 'get_spinorbit_eigenvalues',
                        get_spinorbit_eigenvalues)

    monkeypatch.setattr(gpaw.occupations, 'occupation_numbers',
                        occupation_numbers)


@pytest.fixture
def isolated_filesystem(tmpdir):
    """A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    cwd = os.getcwd()
    os.chdir(str(tmpdir))
    try:
        yield
    finally:
        os.chdir(cwd)


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", '''slow: marks tests as slow (deselect
 with '-m "not slow"')'''
    )
