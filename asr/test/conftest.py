import pytest
import os
import numpy as np
from ase.calculators.calculator import kpts2ndarray, Calculator
from ase.units import Bohr, Ha


class GPAWMock(Calculator):
    """Mock of GPAW calculator.

    Sets up a free electron like gpaw calculator.
    """

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "dipole",
        "magmom",
        "magmoms",
    ]

    default_parameters = {
        "kpts": (4, 4, 4),
        "nelectrons": 4,
        "fermi_level": 0,
        "gap": 0,
        "gridsize": 3,
        "nbands": 8,
        "nspins": 1,
    }

    class Occupations:
        def todict(self):
            return {"name": "fermi-dirac", "width": 0.05}

    occupations = Occupations()

    class Setups(list):
        nvalence = None

    from types import SimpleNamespace

    setups = Setups()
    setups.append(SimpleNamespace(symbol="MySymbol", fingerprint="asdf1234"))

    class WaveFunctions:
        class GridDescriptor:
            pass

        gd = GridDescriptor()

    wfs = WaveFunctions()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ase.io import read

        self.atoms = read("structure.json")

    @classmethod
    def set_property(cls, **kwargs):
        cls.default_parameters.update(kwargs)

    def calculate(self, atoms, *args, **kwargs):
        if atoms is not None:
            self.atoms = atoms
        print(f"Mocked GPAW parameters {self.parameters}")
        if isinstance(self.parameters.nbands, str):
            self.parameters.nbands = int(
                float(self.parameters.nbands[:-1])
                / 100
                * self.parameters.nelectrons
            )
        self.kpts = kpts2ndarray(self.parameters.kpts, atoms)
        icell = atoms.get_reciprocal_cell() * 2 * np.pi * Bohr

        # Simple parabolic band
        n = self.parameters.gridsize
        offsets = np.indices((n, n, n)).T.reshape((n ** 3, 1, 3)) - n // 2
        eps_kn = 0.5 * (np.dot(self.kpts + offsets, icell) ** 2).sum(2).T
        eps_kn.sort()

        # If gap is finite we add another bunch of valence bands
        gap = self.parameters.gap
        eps_kn += gap * 0.5 / Ha
        eps_kn = np.concatenate(
            (-eps_kn[:, ::-1][:, -self.parameters.nelectrons:], eps_kn,),
            axis=1,
        )

        self.setups.nvalence = self.parameters.nelectrons
        self.wfs.gd.cell_cv = atoms.get_cell() / Bohr
        self.eigenvalues = eps_kn[:, : self.parameters.nbands] * Ha
        assert self.eigenvalues.shape[0] == len(self.kpts)
        assert self.eigenvalues.shape[1] == self.parameters.nbands

        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(self.atoms), 3), float),
            "stress": np.zeros((3, 3), float),
            "dipole": np.zeros(3, float),
            "magmom": 0.0,
            "magmoms": np.zeros((len(self.atoms), 3), float),
        }

    def get_fermi_level(self):
        return self.parameters.fermi_level

    def get_eigenvalues(self, kpt, spin=0):
        return self.eigenvalues[kpt]

    def get_k_point_weights(self):
        return [1 / len(self.kpts)] * len(self.kpts)

    def get_ibz_k_points(self):
        return self.kpts.copy()

    def get_number_of_spins(self):
        return self.parameters.nspins

    def get_number_of_bands(self):
        return self.parameters.nbands

    def get_number_of_electrons(self):
        return self.parameters.nelectrons

    def write(self, name):
        from asr.core import write_json

        write_json(name, self.parameters)

    def read(self, name):
        from asr.core import read_json

        class Parameters(dict):
            """Dictionary for parameters.
            
            Special feature: If param is a Parameters instance, then param.xc
            is a shorthand for param['xc'].
            """

            def __getattr__(self, key):
                if key not in self:
                    return dict.__getattribute__(self, key)
                return self[key]

            def __setattr__(self, key, value):
                self[key] = value

        parameters = Parameters(**read_json(name))
        self.parameters = parameters
        from ase.io import read

        self.atoms = read("structure.json")
        self.calculate(self.atoms)

    def set_atoms(self, atoms):
        self.atoms = atoms


@pytest.fixture
def mock_GPAW(monkeypatch):
    import gpaw

    def get_spinorbit_eigenvalues(
        calc,
        bands=None,
        gw_kn=None,
        return_spin=False,
        return_wfs=False,
        scale=1.0,
        theta=0.0,
        phi=0.0,
    ):

        nk = len(calc.get_ibz_k_points())
        nspins = 2
        nbands = calc.get_number_of_bands()
        bands = list(range(nbands))

        e_ksn = np.array(
            [
                [
                    calc.get_eigenvalues(kpt=k, spin=s)[bands]
                    for s in range(nspins)
                ]
                for k in range(nk)
            ]
        )

        s_kvm = np.zeros((nk, 3, nbands), float)
        s_kvm[:, 2, ::2] = 1
        s_kvm[:, 2, ::2] = -1
        e_km = e_ksn.reshape((nk, -1))
        if return_spin:
            return e_km.T, s_kvm
        else:
            return e_km.T

    def occupation_numbers(occ, eps_skn, weight_k, nelectrons):
        fermilevel = GPAWMock.default_parameters.get("fermi_level") / Ha
        f_skn = (eps_skn < fermilevel).astype(float)
        f_skn /= np.prod(f_skn.shape) * nelectrons
        return f_skn, fermilevel, 0.0, 0.0

    from ase.build import bulk

    structure = bulk("Si")
    structure.write("structure.json")
    monkeypatch.setattr(gpaw, "GPAW", GPAWMock)
    monkeypatch.setattr(
        gpaw.spinorbit, "get_spinorbit_eigenvalues", get_spinorbit_eigenvalues
    )
    monkeypatch.setattr(
        gpaw.occupations, "occupation_numbers", occupation_numbers
    )
    return GPAWMock


@pytest.fixture()
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
        "markers",
        """slow: marks tests as slow (deselect
 with '-m "not slow"')""",
    )
