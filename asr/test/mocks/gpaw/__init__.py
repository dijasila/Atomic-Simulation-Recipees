from asr.core import encode_json
from ase.calculators.calculator import kpts2ndarray, Calculator
from ase.units import Bohr, Ha
from ase.symbols import Symbols
import numpy as np
from .mpi import world
from types import SimpleNamespace

__version__ = 'Dummy GPAW version'


class Setups(list):
    nvalence = None

    @property
    def id_a(self):
        for setup in self:
            yield setup.Nv, 'paw', None


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
        "stresses",
        "charges",
        "fermi_level",
        "gap",
        "berry_phases",
    ]

    default_parameters = {
        "kpts": (4, 4, 4),
        "gridsize": 3,
        "nbands": 12,
        "txt": None,
    }

    class Occupations:
        def todict(self):
            return {"name": "fermi-dirac", "width": 0.05}

    occupations = Occupations()

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
        """Calculate properties of atoms and set some necessary instance variables."""
        if atoms is not None:
            self.atoms = atoms

        # These are reasonable instance attributes
        kpts = kpts2ndarray(self.parameters.kpts, atoms)
        self.kpts = kpts
        nbands = self.get_number_of_bands()
        self.eigenvalues = self.get_all_eigenvalues()
        assert self.eigenvalues.shape[0] == len(self.kpts), \
            (self.eigenvalues.shape, self.kpts.shape)
        assert self.eigenvalues.shape[1] == nbands

        # These are unreasonable
        self.setups = self._get_setups()
        self.wfs.kd.nibzkpts = len(kpts)
        self.wfs.kd.weight_k = np.array(self.get_k_point_weights())
        self.setups.nvalence = self.get_number_of_electrons()
        self.wfs.nvalence = self.get_number_of_electrons()
        self.wfs.gd.cell_cv = atoms.get_cell() / Bohr

        self.results = {
            "energy": self._get_potential_energy(),
            "forces": self._get_forces(),
            "stress": self._get_stress(),
            "dipole": self._get_dipole_moment(),
            "magmom": self._get_magmom(),
            "magmoms": self._get_magmoms(),
            "fermi_level": self._get_fermi_level(),
            "gap": self._get_band_gap(),
        }

        if self.parameters.get('txt'):
            data = {'params': self.parameters.copy(),
                    'results': self.results}
            if isinstance(self.parameters.txt, str):
                self.write(self.parameters.txt)
            else:  # Assume that this is a file-descriptor
                data['params'].pop('txt')
                self.parameters.txt.write(encode_json(data))

    def set(self, **kwargs):
        Calculator.set(self, **kwargs)
        self.results = {}

    @property
    def spos_ac(self):
        return self._get_scaled_positions()

    def _get_scaled_positions(self):
        return self.atoms.get_scaled_positions(wrap=True)

    def _get_setups(self):
        setups = Setups()
        for num in self.atoms.get_atomic_numbers():
            setups.append(self._get_setup(num))

        return setups

    def _get_setup_fingerprint(self, element_number):
        return "asdf1234"

    def _get_setup_symbol(self, element_number):
        return str(Symbols([element_number]))

    def _get_setup_nvalence(self, element_number):
        return 1

    def _get_setup(self, element_number):
        setup = SimpleNamespace(symbol=self._get_setup_symbol(element_number),
                                fingerprint=self._get_setup_fingerprint(element_number),
                                Nv=self._get_setup_nvalence(element_number))
        return setup

    def _get_berry_phases(self, dir, spin):
        return np.zeros((10,), float)

    def get_band_gap(self, atoms=None):
        return self.get_property('gap', atoms)

    def _get_band_gap(self):
        """Get band gap."""
        return 0.0

    def _get_dipole_moment(self):
        pos_av = self.atoms.get_positions()
        charges_a = [setup.Nv for setup in self.setups]
        moment = np.dot(charges_a, pos_av)
        return moment

    def get_fermi_level(self, atoms=None):
        return self.get_property('fermi_level', atoms)

    def _get_fermi_level(self):
        return 0.0

    def _get_forces(self):
        return np.zeros((len(self.atoms), 3), float)

    def _get_magmom(self):
        return 0.0

    def _get_magmoms(self):
        return np.zeros((len(self.atoms), 3), float)

    def _get_potential_energy(self):
        return 0.0

    def _get_stress(self):
        return np.zeros((3, 3), float)

    def get_all_eigenvalues(self):
        """Make all eigenvalues for a simple parabolic band."""
        icell = self.atoms.get_reciprocal_cell() * 2 * np.pi * Bohr
        n = self.parameters.gridsize
        offsets = np.indices((n, n, n)).T.reshape((n ** 3, 1, 3)) - n // 2
        eps_kn = 0.5 * (np.dot(self.kpts + offsets, icell) ** 2).sum(2).T
        eps_kn.sort()

        nelectrons = self.get_number_of_electrons()
        gap = self._get_band_gap()
        eps_kn = np.concatenate(
            (-eps_kn[:, ::-1][:, -nelectrons:],
             eps_kn + gap / Ha),
            axis=1,
        )
        nbands = self.get_number_of_bands()
        return eps_kn[:, : nbands] * Ha

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
        return 1

    def get_number_of_bands(self):
        if isinstance(self.parameters.nbands, str):
            return int(
                float(self.parameters.nbands[:-1])
                / 100
                * self.get_number_of_electrons()
            )
        elif self.parameters.nbands < 0:
            return (self.get_number_of_electrons()
                    - self.parameters.nbands)
        else:
            return self.parameters.nbands

    def get_number_of_electrons(self):
        return 4

    def write(self, name, mode=None):
        from asr.core import write_json

        calc = {'atoms': self.atoms,
                'parameters': self.parameters}

        write_json(name, calc)

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

        saved_calc = read_json(name)
        parameters = Parameters(**saved_calc['parameters'])
        self.parameters = parameters
        self.atoms = saved_calc['atoms']
        self.calculate(self.atoms)

    def get_electrostatic_potential(self):
        return np.zeros((20, 20, 20))

    def diagonalize_full_hamiltonian(self, ecut=None):
        pass
