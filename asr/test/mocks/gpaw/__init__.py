from ase.calculators.calculator import kpts2ndarray, Calculator
from ase.units import Bohr, Ha
import numpy as np
from .mpi import world

__version__ = 'Dummy GPAW version'


class GPAW(Calculator):
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
        "gridsize": 3,
        "nbands": 12,
        "nspins": 1,
        "nelectrons": 4,
        "fermi_level": 0,
        "energy": None,
        "forces": None,
        "stress": None,
        "magmom": None,
        "magmoms": None,
        "dipole": np.array([0, 0, 0], float),
        "electrostatic_potential": None,
        "gap": 0,
        "txt": None
    }

    class Occupations:
        def todict(self):
            return {"name": "fermi-dirac", "width": 0.05}

    occupations = Occupations()

    class Setups(list):
        nvalence = None

        id_a = [(0, 'paw', None), ]

    from types import SimpleNamespace

    setups = Setups()
    setups.append(SimpleNamespace(symbol="MySymbol", fingerprint="asdf1234",
                                  Nv=1))

    class WaveFunctions:
        class GridDescriptor:
            pass

        class KPointDescriptor:
            pass

        class BandDescriptor:
            pass

        gd = GridDescriptor()
        bd = BandDescriptor()
        kd = KPointDescriptor()
        nvalence = None

    wfs = WaveFunctions()

    world = world

    def calculate(self, atoms, *args, **kwargs):
        if atoms is not None:
            self.atoms = atoms

        self.spos_ac = atoms.get_scaled_positions(wrap=True)
        kpts = kpts2ndarray(self.parameters.kpts, atoms)
        self.kpts = kpts
        self.wfs.kd.nibzkpts = len(kpts)
        self.wfs.kd.weight_k = np.array(self.get_k_point_weights())
        icell = atoms.get_reciprocal_cell() * 2 * np.pi * Bohr

        # Simple parabolic band
        n = self.parameters.gridsize
        offsets = np.indices((n, n, n)).T.reshape((n ** 3, 1, 3)) - n // 2
        eps_kn = 0.5 * (np.dot(self.kpts + offsets, icell) ** 2).sum(2).T
        eps_kn.sort()

        gap = self.parameters.gap

        eps_kn = np.concatenate(
            (-eps_kn[:, ::-1][:, -self.parameters.nelectrons:],
             eps_kn + gap / Ha),
            axis=1,
        )

        self.setups.nvalence = self.parameters.nelectrons
        self.wfs.nvalence = self.parameters.nelectrons
        self.wfs.gd.cell_cv = atoms.get_cell() / Bohr
        nbands = self.get_number_of_bands()
        self.wfs.bd.nbands = nbands
        self.eigenvalues = eps_kn[:, : nbands] * Ha
        assert self.eigenvalues.shape[0] == len(self.kpts), \
            (self.eigenvalues.shape, self.kpts.shape)
        assert self.eigenvalues.shape[1] == nbands

        self.results = {
            "energy": self.parameters.energy
            or 0.0,
            "forces": self.parameters.forces
            or np.zeros((len(self.atoms), 3), float),
            "stress": self.parameters.stress
            or np.zeros((3, 3), float),
            "dipole": self.parameters.dipole,
            "magmom": self.parameters.magmom or 0.0,
            "magmoms": self.parameters.magmoms
            or np.zeros((len(self.atoms), 3), float),
        }

    def get_fermi_level(self):
        return 0.0

    def get_eigenvalues(self, kpt, spin=0):
        return self.eigenvalues[kpt]

    def get_k_point_weights(self):
        return [1 / len(self.kpts)] * len(self.kpts)

    def get_ibz_k_points(self):
        return self.kpts.copy()

    def get_bz_k_points(self):
        return self.kpts.copy()

    def get_bz_to_ibz_map(self):
        return np.arange(len(self.kpts))

    def get_number_of_spins(self):
        return self.parameters.nspins

    def get_number_of_bands(self):
        if isinstance(self.parameters.nbands, str):
            return int(
                float(self.parameters.nbands[:-1])
                / 100
                * self.parameters.nelectrons
            )
        elif self.parameters.nbands < 0:
            return (self.parameters.nelectrons
                    - self.parameters.nbands)
        else:
            return self.parameters.nbands

    def get_number_of_electrons(self):
        return self.parameters.nelectrons

    def write(self, name, mode=None):
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

    def get_electrostatic_potential(self):
        return (self.parameters.electrostatic_potential
                or np.zeros((20, 20, 20)))

    def diagonalize_full_hamiltonian(self, ecut=None):
        pass
