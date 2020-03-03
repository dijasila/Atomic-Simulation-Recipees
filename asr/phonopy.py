from pathlib import Path

import numpy as np

from ase.parallel import world
from ase.io import read

from asr.core import command, option
from asr.core import write_json


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
    "asr.phonopy",
    requires=["structure.json", "gs.gpw"],
    dependencies=["asr.gs@calculate"],
)
@option("--n", type=int, help="Supercell size")
@option("--d", type=float, help="Displacement size")
@option('-c', '--calculator', help='Calculator params.')
def calculate(n=2, d=0.05,
              calculator={'name': 'gpaw',
                          'mode': {'name': 'pw', 'ecut': 800},
                          'xc': 'PBE',
                          'basis': 'dzp',
                          'kpts': {'density': 6.0, 'gamma': True},
                          'occupations': {'name': 'fermi-dirac',
                                          'width': 0.05},
                          'convergence': {'forces': 1.0e-4},
                          'symmetry': {'point_group': False},
                          'txt': 'phonons.txt',
                          'charge': 0}):
    """Calculate atomic forces used for phonon spectrum."""
    from asr.calculators import get_calculator

    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    # Remove empty files:
    if world.rank == 0:
        for f in Path().glob("phonon.*.json"):
            if f.stat().st_size == 0:
                f.unlink()
    world.barrier()

    atoms = read("structure.json")

    from ase.calculators.calculator import get_calculator_class
    name = calculator.pop('name')
    calc = get_calculator_class(name)(**calculator)

    # Set initial magnetic moments
    from asr.core import is_magnetic

    if is_magnetic():
        gsold = get_calculator()("gs.gpw", txt=None)
        magmoms_m = gsold.get_magnetic_moments()
        atoms.set_initial_magnetic_moments(magmoms_m)

    from asr.core import get_dimensionality

    nd = get_dimensionality()
    if nd == 3:
        supercell = [[n, 0, 0], [0, n, 0], [0, 0, n]]
    elif nd == 2:
        supercell = [[n, 0, 0], [0, n, 0], [0, 0, 1]]
    elif nd == 1:
        supercell = [[n, 0, 0], [0, 1, 0], [0, 0, 1]]

    phonopy_atoms = PhonopyAtoms(symbols=atoms.symbols,
                                 cell=atoms.get_cell(),
                                 scaled_positions=atoms.get_scaled_positions())

    phonon = Phonopy(phonopy_atoms, supercell)

    phonon.generate_displacements(distance=d, is_plusminus=True)
    # displacements = phonon.get_displacements()
    displaced_sc = phonon.get_supercells_with_displacements()

    from ase.atoms import Atoms
    scell = displaced_sc[0]
    atoms_N = Atoms(symbols=scell.get_chemical_symbols(),
                    scaled_positions=scell.get_scaled_positions(),
                    cell=scell.get_cell(),
                    pbc=atoms.pbc)

    for n, cell in enumerate(displaced_sc):
        # Displacement number
        a = n // 2
        # Sign of the displacement
        sign = ["+", "-"][n % 2]

        filename = "phonons.{0}{1}.json".format(a, sign)

        if Path(filename).is_file():
            continue

        atoms_N.set_scaled_positions(cell.get_scaled_positions())
        atoms_N.set_calculator(calc)
        forces = atoms_N.get_forces()

        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]

        write_json(filename, {"force": forces})


def requires():
    return ["results-asr.phonopy@calculate.json"]


def webpanel(row, key_descriptions):
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
    high = "Min. Hessian eig. > -0.01 meV/Ang^2 AND elastic const. > 0"
    medium = "Min. Hessian eig. > -2 eV/Ang^2 AND elastic const. > 0"
    low = "Min. Hessian eig.  < -2 eV/Ang^2 OR elastic const. < 0"
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


@command(
    "asr.phonopy",
    requires=requires,
    webpanel=webpanel,
    dependencies=["asr.phonopy@calculate"],
)
@option("--rc", type=float, help="Cutoff force constants matrix")
def main(rc=None):
    from asr.core import read_json
    from asr.core import get_dimensionality

    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.units import THzToEv

    dct = read_json("results-asr.phonopy@calculate.json")
    atoms = read("structure.json")
    n = dct["__params__"]["n"]
    d = dct["__params__"]["d"]

    nd = get_dimensionality()
    if nd == 3:
        supercell = [[n, 0, 0], [0, n, 0], [0, 0, n]]
        N_c = (n, n, n)
    elif nd == 2:
        supercell = [[n, 0, 0], [0, n, 0], [0, 0, 1]]
        N_c = (n, n, 1)
    elif nd == 1:
        supercell = [[n, 0, 0], [0, 1, 0], [0, 0, 1]]
        N_c = (n, 1, 1)

    phonopy_atoms = PhonopyAtoms(
        symbols=atoms.symbols,
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions(),
    )

    phonon = Phonopy(phonopy_atoms, supercell)

    phonon.generate_displacements(distance=d, is_plusminus=True)
    # displacements = phonon.get_displacements()
    displaced_sc = phonon.get_supercells_with_displacements()

    # for displace in displacements:
    #    print("[Phonopy] %d %s" % (displace[0], displace[1:]))

    set_of_forces = []

    for i, cell in enumerate(displaced_sc):
        # Displacement index
        a = i // 2
        # Sign of the diplacement
        sign = ["+", "-"][i % 2]

        filename = "phonons.{0}{1}.json".format(a, sign)

        forces = read_json(filename)["force"]
        # Number of forces equals to the number of atoms in the supercell
        assert len(forces) == len(atoms) * n ** nd, "Wrong supercell size!"

        set_of_forces.append(forces)

    phonon.produce_force_constants(
        forces=set_of_forces, calculate_full_force_constants=False
    )
    if rc is not None:
        phonon.set_force_constants_zero_with_radius(rc)
    phonon.symmetrize_force_constants()

    q_qc = np.indices(N_c).reshape(3, -1).T / N_c

    omega_kl = np.zeros((len(q_qc), 3 * len(atoms)))
    u_klav = np.zeros((len(q_qc), 3 * len(atoms), len(atoms), 3))

    for q, q_c in enumerate(q_qc):
        omega_l, u_ll = phonon.get_frequencies_with_eigenvectors(q_c)
        omega_kl[q] = omega_l * THzToEv
        u_klav[q] = u_ll.reshape(3 * len(atoms), len(atoms), 3)
        if q_c.any() == 0.0:
            phonon.set_irreps(q_c)
            ob = phonon._irreps
            irreps = []
            for nr, (deg, irr) in enumerate(
                zip(ob._degenerate_sets, ob._ir_labels)
            ):
                irreps += [irr] * len(deg)

    irreps = list(irreps)

    R_cN = lattice_vectors(N_c)
    C_N = phonon.get_force_constants()
    C_N = C_N.reshape(len(atoms), len(atoms), n**nd, 3, 3)
    C_N = C_N.transpose(2, 0, 3, 1, 4)
    C_N = C_N.reshape(n**nd, 3 * len(atoms), 3 * len(atoms))

    eigs = []

    for q_c in q_qc:
        phase_N = np.exp(-2j * np.pi * np.dot(q_c, R_cN))
        C_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * C_N, axis=0)
        eigs.append(np.linalg.eigvalsh(C_q))

    eigs = np.array(eigs)
    mineig = np.min(eigs)

    if mineig < -2:
        dynamic_stability = 1
    elif mineig < -1e-5:
        dynamic_stability = 2
    else:
        dynamic_stability = 3

    results = {'omega_kl': omega_kl,
               'irr_l': irreps,
               'q_qc': q_qc,
               'u_klav': u_klav,
               'minhessianeig': mineig,
               'dynamic_stability_level': dynamic_stability}

    # Next calculate an approximate phonon band structure
    nqpts = 100
    freqs_kl = np.zeros((nqpts, 3 * len(atoms)))
    path = atoms.cell.bandpath(npoints=nqpts, pbc=atoms.pbc)

    for q, q_c in enumerate(path.kpts):
        freqs = phonon.get_frequencies(q_c) * THzToEv
        freqs_kl[q] = freqs

    results['interp_freqs_kl'] = freqs_kl
    results['path'] = path
    results['__key_descriptions__'] = \
        {'minhessianeig': 'KVP: Minimum eigenvalue of Hessian [eV/Ang^2]',
         'dynamic_stability_level': 'KVP: Dynamic stability level'}

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
    from ase.dft.band_structure import BandStructure

    data = row.data.get("results-asr.phonopy.json")
    path = data["path"]
    energies = data["interp_freqs_kl"]
    bs = BandStructure(path=path, energies=energies[None, :, :], reference=0)
    bs.plot(label="Interpolated")

    exact_indices = []
    for q_c in data["q_qc"]:
        diff_kc = path.kpts - q_c
        diff_kc -= np.round(diff_kc)
        inds = np.argwhere(np.all(np.abs(diff_kc) < 1e-3, 1))
        exact_indices.extend(inds.tolist())

    en_exact = np.zeros_like(energies) + np.nan
    for ind in exact_indices:
        en_exact[ind] = energies[ind]

    bs2 = BandStructure(path=path, energies=en_exact[None])
    bs2.plot(
        ax=plt.gca(),
        ls="",
        marker="o",
        color="k",
        emin=np.min(energies * 1.1),
        emax=np.max(energies * 1.1),
        ylabel="Phonon frequencies [eV]",
        label="Exact",
    )
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    main.cli()
