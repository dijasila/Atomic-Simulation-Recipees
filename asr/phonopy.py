"""Phonopy phonon band structure."""
import typing
from pathlib import Path

import numpy as np

from ase.parallel import world
from ase.io import read
from ase.dft.kpoints import BandPath

from asr.core import (command, option, DictStr, ASRResult,
                      read_json, write_json, prepare_result)


def lattice_vectors(N_c):
    """Return lattice vectors for cells in the supercell."""
    # Lattice vectors relevative to the reference cell
    R_cN = np.indices(N_c).reshape(3, -1)
    N_c = np.array(N_c)[:, np.newaxis]
    R_cN += N_c // 2
    R_cN %= N_c
    R_cN -= N_c // 2

    return R_cN


def distance_to_sc(nd, atoms, dist_max):
    if nd >= 1:
        for x in range(2, 20):
            atoms_x = atoms.repeat((x, 1, 1))
            indices_x = [a for a in range(len(atoms_x))]
            dist_x = []
            for a in range(len(atoms)):
                dist = max(atoms_x.get_distances(a, indices_x, mic=True))
                dist_x.append(dist)
            if max(dist_x) > dist_max:
                x_size = x - 1
                break
        supercell = [x_size, 1, 1]
    if nd >= 2:
        for y in range(2, 20):
            atoms_y = atoms.repeat((1, y, 1))
            indices_y = [a for a in range(len(atoms_y))]
            dist_y = []
            for a in range(len(atoms)):
                dist = max(atoms_y.get_distances(a, indices_y, mic=True))
                dist_y.append(dist)
            if max(dist_y) > dist_max:
                y_size = y - 1
                supercell = [x_size, y_size, 1]
                break
    if nd >= 3:
        for z in range(2, 20):
            atoms_z = atoms.repeat((1, 1, z))
            indices_z = [a for a in range(len(atoms_z))]
            dist_z = []
            for a in range(len(atoms)):
                dist = max(atoms_z.get_distances(a, indices_z, mic=True))
                dist_z.append(dist)
            if max(dist_z) > dist_max:
                z_size = z - 1
                supercell = [x_size, y_size, z_size]
                break
    return supercell


@command(
    'asr.phonopy',
    requires=['structure.json', 'gs.gpw']
)
@option('--dftd3', type=bool, help='Enable DFT-D3 for phonon calculations')
@option('--distance', type=float, help='Displacement size')
@option('--dist_max', type=float,
        help='Maximum distance between atoms in the supercell')
@option('--fsname', help='Name for forces file', type=str)
@option('--supercell', nargs=3, type=int,
        help='List of repetitions in lat. vector directions [N_x, N_y, N_z]')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
def calculate(distance: float = 0.05, fsname: str = 'phonons', dftd3: bool = False,
              supercell: typing.List[int] = [1, 1, 1], dist_max: float = 7.0,
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
    # from asr.calculators import get_calculator
    from ase.calculators.dftd3 import DFTD3
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    # Remove empty files:
    if world.rank == 0:
        for f in Path().glob(fsname + '.*.json'):
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
        # gsold = get_calculator()('gs.gpw', txt=None)
        # magmoms_m = gsold.get_magnetic_moments()
        magmoms = np.zeros(len(atoms))
        # magmoms[0:len(atoms)]=2/len(atoms)
        magmoms[0:len(atoms)] = 1
        atoms.set_initial_magnetic_moments(magmoms=magmoms)
        # atoms.set_initial_magnetic_moments(magmoms_m)

    nd = sum(atoms.get_pbc())
    sc = list(map(int, supercell))
    if np.array(sc).any() == 0:
        sc = distance_to_sc(nd, atoms, dist_max)

    if nd == 3:
        supercell = [[sc[0], 0, 0], [0, sc[1], 0], [0, 0, sc[2]]]
    elif nd == 2:
        supercell = [[sc[0], 0, 0], [0, sc[1], 0], [0, 0, 1]]
    elif nd == 1:
        supercell = [[sc[0], 0, 0], [0, 1, 0], [0, 0, 1]]

    phonopy_atoms = PhonopyAtoms(symbols=atoms.symbols,
                                 cell=atoms.get_cell(),
                                 scaled_positions=atoms.get_scaled_positions())
    if is_magnetic():
        # phonopy_atoms.magnetic_moments = atoms.get_initial_magnetic_moments()
        phonopy_atoms.magnetic_moments = atoms.set_initial_magnetic_moments(
            magmoms=magmoms)
    phonon = Phonopy(phonopy_atoms, supercell)

    phonon.generate_displacements(distance=distance, is_plusminus=True)
    displaced_sc = phonon.get_supercells_with_displacements()

    from ase.atoms import Atoms
    scell = displaced_sc[0]
    atoms_N = Atoms(symbols=scell.get_chemical_symbols(),
                    scaled_positions=scell.get_scaled_positions(),
                    cell=scell.get_cell(),
                    pbc=atoms.pbc)

    if is_magnetic():
        # atoms_N.set_initial_magnetic_moments(scell.get_magnetic_moments())
        atoms_N.set_initial_magnetic_moments(magmoms)
    set_of_forces = []

    for n, cell in enumerate(displaced_sc):
        # Displacement number
        a = n // 2
        # Sign of the displacement
        sign = ['+', '-'][n % 2]

        filename = fsname + '.{0}{1}.json'.format(a, sign)

        if Path(filename).is_file():
            forces = read_json(filename)['force']
            set_of_forces.append(forces)
            # Number of forces equals to the number of atoms in the supercell
            assert len(forces) == len(atoms) * np.prod(sc), (
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
    return ['results-asr.phonopy@calculate.json']


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (table, fig,
                                      entry_parameter_description,
                                      describe_entry)

    parameter_description = entry_parameter_description(
        row.data,
        'asr.phonopy@calculate',
        exclude_keys=set(['txt', 'fixdensity', 'verbose', 'symmetry',
                          'idiotproof', 'maxiter', 'hund', 'random',
                          'experimental', 'basis', 'setups']))
    explanation = ('The minimum of the hessian matrix.\n\n'
                   + parameter_description)

    minhessianeig = describe_entry('minhessianeig', description=explanation)

    phonontable = table(row, 'Property', [minhessianeig], key_descriptions)

    panel = {'title': 'Phonons',
             'columns': [[fig('phonon_bs.png')], [phonontable]],
             'plot_descriptions': [{'function': plot_bandstructure,
                                    'filenames': ['phonon_bs.png']}],
             'sort': 3}

    dynstab = row.get('dynamic_stability_phonons')
    high = 'Min. Hessian eig. > -0.01 meV/Ang^2'
    low = 'Min. Hessian eig. <= -0.01 meV/Ang^2'
    row = ['Dynamical (phonons)',
           '<a href="#" data-toggle="tooltip" data-html="true" '
           + 'title="LOW: {}&#13;HIGH: {}">{}</a>'.format(
               low, high, dynstab.upper())]

    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Stability', 'Category'],
                             'rows': [row]}]],
               'sort': 2}
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
    dynamic_stability_phonons: int

    key_descriptions = dict(
        omega_kl='Phonon frequencies.',
        minhessianeig='Minimum eigenvalue of Hessian [`eV/Ang^2`]',
        eigs_kl='Dynamical matrix eigenvalues.',
        q_qc='List of momenta consistent with supercell.',
        phi_anv='Force constants.',
        u_klav='Phonon modes.',
        irr_l='Phonon irreducible representations.',
        path='Phonon bandstructure path.',
        dynamic_stability_phonons='Phonon dynamic stability (low/high)',
    )

    formats = {"ase_webpanel": webpanel}


@prepare_result
class HessResult(ASRResult):
    minhessianeig: float

    key_descriptions: typing.Dict[str, str] = dict(
        minhessianeig='Minimum eigenvalue of Hessian [`eV/Ang^2`]'
    )


@command(
    'asr.phonopy',
    requires=requires,
    returns=Result,
    dependencies=['asr.phonopy@calculate'],
)
@option('--rc', type=float, help='Cutoff force constants matrix')
@option('--nqpts', type=int, help='Number of points in the path')
def main(rc: float = None, nqpts: int = 100):
    import phonopy
    from phonopy.units import THzToEv

    calculateresult = read_json('results-asr.phonopy@calculate.json')
    dist_max = calculateresult.metadata.params['dist_max']
    sc = calculateresult.metadata.params['supercell']
    sc = list(map(int, sc))
    atoms = read('structure.json')
    nd = sum(atoms.get_pbc())

    if np.array(sc).any() == 0:
        sc = distance_to_sc(nd, atoms, dist_max)

    phonon = phonopy.load('phonopy_params.yaml')

    if rc is not None:
        phonon.set_force_constants_zero_with_radius(rc)
        phonon.symmetrize_force_constants()

    path = atoms.cell.bandpath(npoints=nqpts, pbc=atoms.pbc)

    omega_kl = np.zeros((nqpts, 3 * len(atoms)))

    # Calculating phonon frequencies along a path in the BZ
    for q, q_c in enumerate(path.kpts):
        omega_l = phonon.get_frequencies(q_c)
        omega_kl[q] = omega_l * THzToEv * 1000

    R_cN = lattice_vectors(sc)
    C_N = phonon.get_force_constants()
    C_N = C_N.reshape(len(atoms), len(atoms), np.prod(sc), 3, 3)
    C_N = C_N.transpose(2, 0, 3, 1, 4)
    C_N = C_N.reshape(np.prod(sc), 3 * len(atoms), 3 * len(atoms))

    # Calculating hessian and eigenvectors at high symmetry points of the BZ
    eigs_kl = []
    q_qc = list(path.special_points.values())
    u_klav = np.zeros((len(q_qc), 3 * len(atoms), len(atoms), 3), dtype=complex)

    for q, q_c in enumerate(q_qc):
        phase_N = np.exp(-2j * np.pi * np.dot(q_c, R_cN))
        C_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * C_N, axis=0)
        eigs_kl.append(np.linalg.eigvalsh(C_q))
        _, u_ll = phonon.get_frequencies_with_eigenvectors(q_c)
        u_klav[q] = u_ll.T.reshape(3 * len(atoms), len(atoms), 3)
        if q_c.any() == 0.0:
            phonon.set_irreps(q_c)
            irreps = []
            try:
                ob = phonon._irreps
                for nr, (deg, irr) in enumerate(
                        zip(ob._degenerate_sets, ob._ir_labels)):
                    irreps += [irr] * len(deg)
            except AttributeError:
                continue

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

    data = row.data.get('results-asr.phonopy.json')
    if data is None:
        return

    omega_kl = data['omega_kl']
    gamma = omega_kl[0]
    fig = plt.figure(figsize=(6.4, 3.9))
    ax = fig.gca()

    x0 = -0.0005  # eV
    for x, color in [(gamma[gamma < x0], 'r'), (gamma[gamma >= x0], 'b')]:
        if len(x) > 0:
            markerline, _, _ = ax.stem(
                x * 1000,
                np.ones_like(x),
                bottom=-1,
                markerfmt=color + 'o',
                linefmt=color + '-',
            )
            plt.setp(markerline, alpha=0.4)
    ax.set_xlabel(r'phonon frequency at $\Gamma$ [meV]')
    ax.axis(ymin=0.0, ymax=1.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_bandstructure(row, fname):
    from matplotlib import pyplot as plt
    from ase.spectrum.band_structure import BandStructure

    data = row.data.get('results-asr.phonopy.json')
    path = data['path']
    energies = data['omega_kl']
    bs = BandStructure(path=path, energies=energies[None, :, :], reference=0)
    bs.plot(
        color="k",
        emin=np.min(energies * 1.1),
        emax=np.max(energies * 1.1),
        ylabel='Phonon frequencies [meV]',
    )

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


@command(
    'asr.phonopy',
    requires=requires,
    webpanel=webpanel,
    dependencies=['asr.phonopy'],
)
@option('-q', '--momentum', nargs=3, type=float,
        help='Phonon momentum')
@option('-s', '--supercell', nargs=3, type=int,
        help='Supercell sizes')
@option('-m', '--mode', type=int, help='Mode index')
@option('-a', '--amplitude', type=float,
        help='Maximum distance an atom will be displaced')
@option('--nimages', type=int, help='Mode index')
def write_mode(momentum: typing.List[float] = [0, 0, 0], mode: int = 0,
               supercell: typing.List[int] = [1, 1, 1], amplitude: float = 0.1,
               nimages: int = 30):

    from ase.io.trajectory import Trajectory

    q_c = momentum
    atoms = read('structure.json')
    data = read_json('results-asr.phonopy.json')
    u_klav = data['u_klav']
    q_qc = data['q_qc']
    diff_kc = np.array(list(q_qc)) - q_c
    diff_kc -= np.round(diff_kc)
    ind = np.argwhere(np.all(np.abs(diff_kc) < 1e-2, 1))[0, 0]

    # Repeat atoms
    repeat_c = supercell
    newatoms = atoms * repeat_c
    # Here `Na` refers to a composite unit cell/atom dimension
    pos_Nav = newatoms.get_positions()
    # Total number of unit cells
    N = np.prod(repeat_c)

    # Corresponding lattice vectors R_m
    R_cN = np.indices(repeat_c).reshape(3, -1)

    # Bloch phase
    phase_N = np.exp(2j * np.pi * np.dot(q_c, R_cN))
    phase_Na = phase_N.repeat(len(atoms))
    m_Na = newatoms.get_masses()

    # Repeat and multiply by Bloch phase factor
    mode_av = u_klav[ind, mode]
    n_a = np.linalg.norm(mode_av, axis=1)
    mode_av /= np.max(n_a)
    mode_Nav = np.vstack(N * [mode_av]) * phase_Na[:, np.newaxis] \
                                        * amplitude / m_Na[:, np.newaxis]

    filename = 'mode-q-{}-{}-{}-mode-{}.traj'.format(q_c[0], q_c[1], q_c[2], mode)
    traj = Trajectory(filename, 'w')

    for x in np.linspace(0, 2 * np.pi, nimages, endpoint=False):
        newatoms.set_positions((pos_Nav + np.exp(1.j * x) * mode_Nav).real)
        traj.write(newatoms)

    traj.close()


if __name__ == '__main__':
    main.cli()
