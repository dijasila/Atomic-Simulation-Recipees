from asr.core import encode_json
from ase.calculators.calculator import kpts2ndarray, Calculator
from ase.units import Bohr, Ha
from ase.symbols import Symbols
import numpy as np
from .mpi import world
from types import SimpleNamespace


class Parameters(dict):
    """Dictionary for parameters.

    Special feature: If param is a Parameters instance, then param.xc
    is a shorthand for param['xc'].
    """

    def __getattr__(self, key):
        """Get attribute."""
        if key not in self:
            return dict.__getattribute__(self, key)
        return self[key]

    def __setattr__(self, key, value):
        """Set an attribute."""
        self[key] = value


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


class Occupations:
    def todict(self):
        return {"name": "fermi-dirac", "width": 0.05}


class Setups(list):
    nvalence = None

    @property
    def id_a(self):
        for setup in self:
            yield setup.Nv, 'paw', None


class ASRCalculator(Calculator):
    """Mock of a generic ASE calculator.

    This test calculator is specifically designed to look like the
    GPAW calculator, and to make additional monkey patching easy. Each
    of the public methods have an identical private method prefaced
    with underscore that can be easily patch using the pytest
    mocker.patch functionality. For example,
    ASRCalculator.get_fermi_level has an identical
    ASRCalculator._get_fermi_level method that can be easily mocked
    up.

    By default, the eigenvalues set up the calculator has some valence
    bands with negative curvature and some conduction bands with
    positive curvature. Furthermore, if the return value of
    _get_band_gap is non-zero the valence and conduction bands will be
    separated by a band gap. This provides an easy way to manipulate
    the properties of a test material.
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

    occupations = Occupations()

    wfs = WaveFunctions()

    world = world

    def __init__(self, txt=None, **kwargs):
        # We don't provide txt to the super constructor because we don't
        # want that to be a "parameter" and then mess with trajectory
        # writing whenever it contains an actual file object.
        super().__init__(**kwargs)

    def calculate(self, atoms, *args, **kwargs):
        """Calculate properties of atoms and set some necessary instance variables.

        This is the main method that calculates the energy, forces and
        other properties of the given structure. This method should
        not be mocked. In stead mock the implementation of the
        specific property you need to amend.

        """
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
        self.setups.nvalence = self.get_number_of_valence_electrons()
        self.wfs.nvalence = self.get_number_of_valence_electrons()
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

        # TODO: Fix this hack
        ASRCalculator._gap = self.results["gap"]
        ASRCalculator._fermi_level = self.results["fermi_level"]
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
        """Get scaled positions."""
        return self.atoms.get_scaled_positions(wrap=True)

    def _get_setups(self):
        """Get all setups."""
        setups = Setups()
        for num in self.atoms.get_atomic_numbers():
            setups.append(self._get_setup(num))

        return setups

    def _get_setup_fingerprint(self, element_number):
        """Get specific setup fingerprint.

        Parameters
        ----------
        element_number : int
            Atomic number of element.

        Returns
        -------
        str

        """
        return "asdf1234"

    def _get_setup_symbol(self, element_number):
        """Get setup symbol."""
        return str(Symbols([element_number]))

    def _get_setup_nvalence(self, element_number):
        """Get number of valence electrons.

        This also dynamically controls the number of valence bands.
        """
        return 1

    def _get_setup(self, element_number):
        """Get specific setup.

        Parameters
        ----------
        element_number : int
            Atomic number of element.

        Returns
        -------
        SimpleNamespace
            A SimpleNamespace that resembles a GPAW Setups class.

        """
        setup = SimpleNamespace(symbol=self._get_setup_symbol(element_number),
                                fingerprint=self._get_setup_fingerprint(
                                    element_number),
                                Nv=self._get_setup_nvalence(element_number))
        return setup

    def _get_berry_phases(self, dir, spin):
        """Return berry phases.

        Parameters
        ----------
        dir : int
            Which direction to return berry phases along.
        spin : int
            Spin channel.

        Returns
        -------
        np.ndarray
            Berry phases along a specific axis

        """
        return np.zeros((10,), float)

    def _get_band_gap(self):
        """Get band gap."""
        return 0.0

    def _get_dipole_moment(self):
        pos_av = self.atoms.get_positions()
        charges_a = [setup.Nv for setup in self.setups]
        moment = np.dot(charges_a, pos_av)
        return moment

    def get_fermi_level(self, atoms=None):
        """Get cached fermi level."""
        return self.get_property('fermi_level', atoms)

    def _get_fermi_level(self):
        """Get formi level."""
        return 0.0

    def _get_forces(self):
        """Get atomic forces."""
        return np.zeros((len(self.atoms), 3), float)

    def _get_magmom(self):
        """Get total magnetic moment."""
        return 0.0

    def _get_magmoms(self):
        """Get atomic magnetic moments."""
        return np.ones(len(self.atoms), float)

    def _get_potential_energy(self):
        """Get potential energy."""
        return 0.0

    def _get_stress(self):
        """Get cell stress."""
        return np.zeros((3, 3), float)

    def get_all_eigenvalues(self):
        """Get all eigenvalues.

        Constructs all eigenvalues for the test calculator. The
        valence bands will have a negative curvature and the number of
        valence bands are determined by
        :py:meth:`ASRCalculator.get_number_of_valence_electrons`. Conduction
        bands have positive curvature and are separated from the
        valence bands by a bandgap. The band gap is obtained from
        :py:meth:`ASRCalculator._get_band_gap`.

        """
        icell = self.atoms.cell.reciprocal() * 2 * np.pi * Bohr
        n = self.parameters.gridsize
        offsets = np.indices((n, n, n)).T.reshape((n ** 3, 1, 3)) - n // 2
        eps_kn = 0.5 * (np.dot(self.kpts + offsets, icell) ** 2).sum(2).T
        eps_kn.sort()

        nvalence = self.get_number_of_valence_electrons()
        nvalencebands = nvalence // 2
        gap = self._get_band_gap()
        eps_kn = np.concatenate(
            (-eps_kn[:, ::-1][:, -nvalencebands:],
             eps_kn + gap / Ha),
            axis=1,
        )
        nbands = self.get_number_of_bands()
        return eps_kn[:, :nbands] * Ha

    def get_eigenvalues(self, kpt, spin=0):
        """Return the eigenvalues of a specific k-point.

        Parameters
        ----------
        kpt : int
            K-point index.
        spin : int
            Spin channel.

        """
        return self.eigenvalues[kpt].copy()

    def get_k_point_weights(self):
        """Get all k-point weights."""
        return [1 / len(self.kpts)] * len(self.kpts)

    def get_ibz_k_points(self):
        """Get an array of all irreducible k-points."""
        return self.kpts.copy()

    def get_bz_k_points(self):
        """Get an array of all k-points."""
        return self.kpts.copy()

    def get_bz_to_ibz_map(self):
        """Get BZ to IBZ map."""
        return np.arange(len(self.kpts))

    def get_number_of_spins(self):
        """Get number of spins in calculation."""
        return 1

    def get_number_of_bands(self):
        """Get total number of bands in calculation."""
        if isinstance(self.parameters.nbands, str):
            return int(
                float(self.parameters.nbands[:-1])
                / 100
                * self.get_number_of_valence_electrons()
            )
        elif self.parameters.nbands < 0:
            return int((self.get_number_of_valence_electrons() / 2
                        - self.parameters.nbands))
        else:
            return self.parameters.nbands

    def get_number_of_conduction_electrons(self):
        """Get number of conduction electrons per unit cell."""
        fermi_level = self._get_fermi_level()
        if not fermi_level > 0.0:
            return 0
        nkpts = len(self.get_bz_k_points())
        return (np.sum(self.eigenvalues < fermi_level) * 2 / nkpts
                - self.get_number_of_valence_electrons())

    def get_number_of_electrons(self):
        """Get number of electrons."""
        return self.get_number_of_valence_electrons() + \
            self.get_number_of_conduction_electrons()

    def get_number_of_valence_electrons(self):
        """Get number of valence electrons.

        The number of valence electrons exclude any extra doping there
        might exist due to any additional doping.

        """
        return 4

    def write(self, name, mode=None):
        """Write calculator to file."""
        from asr.core import write_json
        from copy import deepcopy

        # We have to do a deep copy because the current version of
        # ase.calculators.calculator.KPoints overwrites the __dict__
        # attribute of KPoints upon writing the first time.
        calc = {
            'atoms': self.atoms,
            'parameters': deepcopy(self.parameters)
        }

        write_json(name, calc)

    def read(self, name):
        """Read calculator from file."""
        from asr.core import read_json

        saved_calc = read_json(name)
        parameters = Parameters(**saved_calc['parameters'])
        self.parameters = parameters
        self.atoms = saved_calc['atoms']
        self.calculate(self.atoms)

    def get_electrostatic_potential(self):
        """Get electrostatic potential."""
        return np.zeros((20, 20, 20))

    def diagonalize_full_hamiltonian(self, ecut=None):
        """Diagonalize full Hamiltonian."""
        pass

    def dos(self, soc=False, theta=0.0, phi=0.0, shift_fermi_level=True):
        return DOSCalculator(self.get_fermi_level())

    def fixed_density(self, **kwargs):
        return self


class DOSCalculator:
    def __init__(self, fermi_level):
        self.fermi_level = fermi_level

    def raw_dos(self, energies, spin=None, width=0.1):
        return np.ones_like(energies)

    def raw_pdos(self, energies, a, l, m=None, spin=None, width=0.1):
        return np.ones_like(energies)
