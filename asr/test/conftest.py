import pytest
import os
import numpy as np
from ase.calculators.calculator import Calculator, kpts2ndarray
from ase.units import Bohr, Ha

from pathlib import Path


class FreeElectronsGPAW(Calculator):
    """Free-electron band calculator.

    Parameters:

    nvalence: int
        Number of electrons
    kpts: dict
        K-point specification.

    Example:

    >>> calc = FreeElectrons(nvalence=1, kpts={'path': 'GXL'})
    """

    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {'kpts': np.zeros((1, 3)),
                          'nvalence': 0.0,
                          'nbands': 20,
                          'gridsize': 7}

    class Setups(list):
        nvalence = 20

    from types import SimpleNamespace
    setups = Setups()
    setups.append(SimpleNamespace(symbol='MySymbol', fingerprint='asdf1234'))

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms)
        self.kpts = kpts2ndarray(self.parameters.kpts, atoms)
        icell = atoms.get_reciprocal_cell() * 2 * np.pi * Bohr
        n = self.parameters.gridsize
        offsets = np.indices((n, n, n)).T.reshape((n**3, 1, 3)) - n // 2
        eps = 0.5 * (np.dot(self.kpts + offsets, icell)**2).sum(2).T
        eps.sort()
        if isinstance(self.parameters.nbands, str) and \
           self.parameters.nbands[-1] == '%':
            self.parameters.nbands = \
                int(float(self.parameters.nbands[:-1]) / 100 * 20)
        print(self.parameters.nbands)
        self.eigenvalues = eps[:, :self.parameters.nbands] * Ha
        self.results = {'energy': 0.0,
                        'forces': np.zeros((len(self.atoms), 3), float),
                        'stress': np.zeros((3, 3), float)}

    def get_eigenvalues(self, kpt, spin=0):
        assert spin == 0
        return self.eigenvalues[kpt].copy()

    def get_fermi_level(self):
        v = self.atoms.get_volume() / Bohr**3
        kF = (self.parameters.nvalence / v * 3 * np.pi**2)**(1 / 3)
        return 0.5 * kF**2 * Ha

    def get_ibz_k_points(self):
        return self.kpts.copy()

    def get_number_of_spins(self):
        return 1

    def write(self, name):
        Path(name).write_text('Test calculation')

    def read(self, name):
        Path(name).read_text('Test calculation')


@pytest.fixture
def mock_GPAW_freeelectrons(monkeypatch):
    import gpaw
    monkeypatch.setattr(gpaw, 'GPAW', FreeElectronsGPAW)


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
