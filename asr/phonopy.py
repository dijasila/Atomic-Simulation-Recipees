"""Phonopy phonon band structure."""
import typing
from pathlib import Path

import numpy as np

from ase.atoms import Atoms
from ase.parallel import world
from ase.io import read
from ase.dft.kpoints import BandPath

from asr.core import (command, option, DictStr, ASRResult,
                      read_json, write_json, prepare_result)

import phonopy


def lattice_vectors(N_c):
    """Return lattice vectors for cells in the supercell."""
    # Lattice vectors relevative to the reference cell
    R_cN = np.indices(N_c).reshape(3, -1)
    N_c = np.array(N_c)[:, np.newaxis]
    R_cN += N_c // 2
    R_cN %= N_c
    R_cN -= N_c // 2

    return R_cN


@command(
    'asr.phonopy',
    requires=['structure.json', 'gs.gpw'],
    dependencies=['asr.gs@calculate'],
)
@option('--dftd3', type=bool, help='Enable DFT-D3 for phonon calculations')
@option('--displacement', type=float, help='Displacement size')
@option('--forcesfolder', help='Name for forces file', type=str)
@option('--supercell', nargs=3, type=int,
        help='List of repetitions in lat. vector directions [N_x, N_y, N_z]')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
def calculate(dftd3: bool = False,
              displacement: float = 0.05,
              forcesfolder: str = 'phonons',
              supercell: typing.List[int] = (2, 2, 2),
              calculator: dict = {'name': 'gpaw',
                                  'mode': {'name': 'pw', 'ecut': 800},
                                  'xc': 'PBE',
                                  'basis': 'dzp',
                                  'kpts': {'density': 6.0, 'gamma': True},
                                  'occupations': {'name': 'fermi-dirac',
                                                  'width': 0.05},
                                  'convergence': {'forces': 1.0e-4},
                                  'symmetry': {'point_group': False},
                                  'txt': 'phonons.txt',
                                  'charge': 0}) -> ASRResult:
    """Calculate atomic forces used for phonon spectrum."""
    from asr.calculators import get_calculator
    from ase.calculators.dftd3 import DFTD3
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    # Remove empty files:
    if world.rank == 0:
        for f in Path().glob(forcesfolder + '/phonons.*.json'):
            if f.stat().st_size == 0:
                f.unlink()
    world.barrier()

    atoms = read('structure.json')

    from ase.calculators.calculator import get_calculator_class
    name = calculator.pop('name')
    calc = get_calculator_class(name)(**calculator)
    if dftd3:
        calc = DFTD3(dft=calc, cutoff=60)

    # Set initial magnetic moments
    from asr.utils import is_magnetic

    if is_magnetic():
        gsold = get_calculator()('gs.gpw', txt=None)
        magmoms_m = gsold.get_magnetic_moments()
        atoms.set_initial_magnetic_moments(magmoms_m)

    ndim = sum(atoms.pbc)
    assert all(atoms.pbc[:ndim])

    supercell_triple = np.ones(3)
    supercell_triple[:ndim] = supercell
    supercell_matrix = np.diag(supercell_triple)

    phonopy_atoms = PhonopyAtoms(symbols=atoms.symbols,
                                 cell=atoms.get_cell(),
                                 scaled_positions=atoms.get_scaled_positions())
    if is_magnetic():
        phonopy_atoms.magnetic_moments = atoms.get_initial_magnetic_moments()

    phonon = Phonopy(phonopy_atoms, supercell_matrix)

    phonon.generate_displacements(distance=displacement, is_plusminus=True)
    displaced_sc = phonon.get_supercells_with_displacements()

    scell = displaced_sc[0]
    atoms_N = Atoms(symbols=scell.get_chemical_symbols(),
                    scaled_positions=scell.get_scaled_positions(),
                    cell=scell.get_cell(),
                    pbc=atoms.pbc)

    if is_magnetic():
        atoms_N.set_initial_magnetic_moments(scell.get_magnetic_moments())

    set_of_forces = []

    folder = Path("phonons")
    folder.mkdir(parents=True, exist_ok=True)

    for n, cell in enumerate(displaced_sc):
        # Displacement number
        a = n // 2
        # Sign of the displacement
        sign = ['+', '-'][n % 2]

        filename = forcesfolder + '/phonons.{0}{1}.json'.format(a, sign)

        if Path(filename).is_file():
            forces = read_json(filename)['force']
            set_of_forces.append(forces)
            # Number of forces equals to the number of atoms in the supercell
            assert len(forces) == len(atoms) * np.prod(supercell), (
                'Wrong supercell size!')
            continue

        atoms_N.set_scaled_positions(cell.get_scaled_positions())
        atoms_N.calc = calc
        forces = atoms_N.get_forces()

        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)
        write_json(filename, {'force': forces})

    phonon.produce_force_constants(
        forces=set_of_forces,
        calculate_full_force_constants=False)
    phonon.symmetrize_force_constants()

    phonon.save(settings={'force_constants': True})


def requires():
    return ["results-asr.phonopy@calculate.json"]


def webpanel(result, row, key_descriptions):
    from asr.database.browser import table, fig

    phonontable = table(row, "Property", ["minhessianeig"], key_descriptions)

    panel = {
        "title": "Phonon bandstructure",
        "columns": [[fig("phonon_bs.png")], [phonontable]],
        "plot_descriptions": [
            {"function": plot_bandstructure, "filenames": ["phonon_bs.png"]}
        ],
        "sort": 3,
    }

    dynstab = row.get("dynamic_stability_level")
    stabilities = {1: "low", 2: "medium", 3: "high"}
    high = "Minimum eigenvalue of Hessian > -0.01 meV/Å² AND elastic const. > 0"
    medium = "Minimum eigenvalue of Hessian > -2 eV/Å² AND elastic const. > 0"
    low = "Minimum eigenvalue of Hessian < -2 eV/Å² OR elastic const. < 0"
    row = [
        "Phonons",
        '<a href="#" data-toggle="tooltip" data-html="true" '
        + 'title="LOW: {}&#13;MEDIUM: {}&#13;HIGH: {}">{}</a>'.format(
            low, medium, high, stabilities[dynstab].upper()
        ),
    ]

    summary = {
        "title": "Summary",
        "columns": [
            [
                {
                    "type": "table",
                    "header": ["Stability", "Category"],
                    "rows": [row],
                }
            ]
        ],
    }
    return [panel, summary]


@prepare_result
class Result(ASRResult):
    omega_kl: typing.List[typing.List[float]]
    minhessianeig: float
    eigs_kl: typing.List[typing.List[complex]]
    q_qc: typing.List[typing.Tuple[float, float, float]]
    phi_anv: typing.List[typing.List[typing.List[float]]]
    u_klav: typing.List[typing.List[float]]
    irr_l: typing.List[str]
    path: BandPath
    dynamic_stability_phonons: str

    key_descriptions = {
        "omega_kl": "Phonon frequencies.",
        "minhessianeig": "Minimum eigenvalue of Hessian [`eV/Å²`]",
        "eigs_kl": "Dynamical matrix eigenvalues.",
        "q_qc": "List of momenta consistent with supercell.",
        "phi_anv": "Force constants.",
        "u_klav": "Phonon modes.",
        "irr_l": "Phonon irreducible representations.",
        "path": "Phonon bandstructure path.",
        "dynamic_stability_phonons": "Phonon dynamic stability (low/high)",
    }

    formats = {"ase_webpanel": webpanel}


@command(
    "asr.phonopy",
    requires=requires,
    returns=Result,
    dependencies=["asr.phonopy@calculate"],
)
@option("--cutoff", type=float, help="Cutoff for the force constants matrix")
@option("--nac", type=bool, help="Non-analytical term correction")
@option("--nqpts", type=int, help="Number of q-points along the path")
def main(cutoff: float = None, nac: bool = False, nqpts: int = 400) -> Result:
    from phonopy.units import THzToEv

    from math import pi

    calculateresult = read_json("results-asr.phonopy@calculate.json")
    atoms = read("structure.json")
    params = calculateresult.metadata.params
    supercell = list(map(int, params["supercell"]))

    phonon = phonopy.load('phonopy_params.yaml')

    if nac:
        results_borncharges = "results-asr.borncharges.json"
        if not Path(results_borncharges).is_file():
            print("Calculate born charges first!")

        Z_avv = read_json(results_borncharges)["Z_avv"]

        results_polarizability = "results-asr.polarizability.json"
        if not Path(results_polarizability).is_file():
            print("Calculate polarizability first!")

        e_v = []
        for mu in ["x", "y", "z"]:
            e = 4 * pi * read_json(results_polarizability)[f"alpha{mu}_el"] + 1
            e_v.append(e)

        nac_params = {"born": Z_avv, "factor": 14.4, "dielectric": np.diag(e_v)}
        phonon.set_nac_params(nac_params)

    if cutoff is not None:
        phonon.set_force_constants_zero_with_radius(cutoff)
    phonon.symmetrize_force_constants()

    path = atoms.cell.bandpath(npoints=nqpts, pbc=atoms.pbc)

    omega_kl = np.zeros((nqpts, 3 * len(atoms)))

    # Calculating phonon frequencies along a path in the BZ
    for q, q_c in enumerate(path.kpts):
        if (nac and q_c.any() == 0):
            print('Avoid Gamma point if NAC is included!')
            q_c = (q_c - path.kpts[q + 1]) / 2

        omega_l = phonon.get_frequencies(q_c)
        omega_kl[q] = omega_l * THzToEv

    R_cN = lattice_vectors(supercell)
    C_N = phonon.get_force_constants()
    C_N = C_N.reshape(len(atoms), len(atoms), np.prod(supercell), 3, 3)
    C_N = C_N.transpose(2, 0, 3, 1, 4)
    C_N = C_N.reshape(np.prod(supercell), 3 * len(atoms), 3 * len(atoms))

    # Calculating hessian and eigenvectors at high symmetry points of the BZ
    eigs_kl = []
    q_qc = np.indices(supercell).reshape(3, -1).T / supercell
    u_klav = np.zeros((len(q_qc), 3 * len(atoms), len(atoms), 3), dtype=complex)

    for q, q_c in enumerate(q_qc):
        phase_N = np.exp(-2j * np.pi * np.dot(q_c, R_cN))
        C_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * C_N, axis=0)
        eigs_kl.append(np.linalg.eigvalsh(C_q))
        _, u_ll = phonon.get_frequencies_with_eigenvectors(q_c)
        u_klav[q] = u_ll.reshape(3 * len(atoms), len(atoms), 3)
        if q_c.any() == 5.0:
            phonon.set_irreps(q_c)
            ob = phonon._irreps
            irreps = []
            for nr, (deg, irr) in enumerate(
                zip(ob._degenerate_sets, ob._ir_labels)
            ):
                irreps += [irr] * len(deg)
    irreps = []
    irreps = list(irreps)

    eigs_kl = np.array(eigs_kl)
    mineig = np.min(eigs_kl)

    if mineig < -0.01:
        dynamic_stability = 'low'
    else:
        dynamic_stability = 'high'

    phi_anv = phonon.get_force_constants()

    results = {'omega_kl': omega_kl,
               'eigs_kl': eigs_kl,
               'phi_anv': phi_anv,
               'irr_l': irreps,
               'q_qc': q_qc,
               'path': path,
               'u_klav': u_klav,
               'minhessianeig': mineig,
               'dynamic_stability_phonons': dynamic_stability}

    return results


def plot_phonons(row, fname):
    import matplotlib.pyplot as plt

    data = row.data.get("results-asr.phonopy.json")
    if data is None:
        return

    omega_kl = data["omega_kl"]
    gamma = omega_kl[0]
    fig = plt.figure(figsize=(6.4, 3.9))
    ax = fig.gca()

    x0 = -0.0005  # eV
    for x, color in [(gamma[gamma < x0], "r"), (gamma[gamma >= x0], "b")]:
        if len(x) > 0:
            markerline, _, _ = ax.stem(
                x * 1000,
                np.ones_like(x),
                bottom=-1,
                markerfmt=color + "o",
                linefmt=color + "-",
            )
            plt.setp(markerline, alpha=0.4)
    ax.set_xlabel(r"phonon frequency at $\Gamma$ [meV]")
    ax.axis(ymin=0.0, ymax=1.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_bandstructure(row, fname):
    from matplotlib import pyplot as plt
    from ase.spectrum.band_structure import BandStructure

    data = row.data.get("results-asr.phonopy.json")
    path = data["path"]
    energies = data["omega_kl"]
    bs = BandStructure(path=path, energies=energies[None, :, :], reference=0)
    bs.plot(
        color="k",
        emin=np.min(energies * 1.1),
        emax=np.max(energies * 1.1),
        ylabel="Phonon frequencies [meV]",
    )

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


@command(
    'asr.phonopy',
    requires=requires,
    dependencies=['asr.phonopy'],
)
@option('--momentum', nargs=3, type=float,
        help='Phonon momentum')
@option('--supercell', nargs=3, type=int,
        help='Supercell sizes')
@option('--mode', type=int, help='Mode index')
@option('--amplitude', type=float,
        help='Amplitude of oscillation in amu^1/2 A')
@option('--nimages', type=int, help='Mode index')
def write_mode(momentum: typing.List[float] = (0., 0., 0.), mode: int = 0,
               supercell: typing.List[int] = (1, 1, 1), amplitude: float = 1,
               nimages: int = 15) -> None:

    from ase.io.trajectory import Trajectory

    from math import sqrt

    atoms = read('structure.json')
    Na = len(atoms)

    phonon = phonopy.load('phonopy_params.yaml')

    amplitudes2_n = np.linspace(-amplitude, amplitude, 16, endpoint=True)
    amplitudes_n = np.hstack((amplitudes2_n[:-1], amplitudes2_n[::-1]))
    q_c = momentum

    filename = 'mode-q-{}-{}-{}-mode-{}.traj'.format(q_c[0], q_c[1], q_c[2], mode)

    with Trajectory(filename, 'w') as traj:
        for amplitude in amplitudes_n:

            phonon.set_modulations(dimension=supercell,
                                   phonon_modes=[[momentum,
                                                  mode,
                                                  amplitude,
                                                  0]])

            delta_R_ai = phonon.get_modulations_and_supercell()[0][0] * sqrt(Na)
            scell = phonon.get_modulations_and_supercell()[1]

            atoms_q = Atoms(symbols=scell.get_chemical_symbols(),
                            positions=scell.get_positions() + delta_R_ai,
                            cell=scell.get_cell(),
                            pbc=atoms.pbc)

            traj.write(atoms_q)

    traj.close()


if __name__ == "__main__":
    main.cli()
