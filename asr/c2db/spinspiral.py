from ase import Atoms
from ase.dft.kpoints import BandPath
from asr.core import (
    instruction, atomsopt, option,
    argument, ASRResult, prepare_result
)

import numpy as np


@instruction(module='asr.c2db.spinspiral')
@atomsopt
@argument('q_c', type=str)
@option('--kptdensity', help='Kpoint density', type=float)
@option('--ecut', help='Energy cutoff', type=float)
def calculate(atoms: Atoms, q_c: str = "[0 0 0]", ecut: float = 600,
              kptdensity: float = 4.0) -> ASRResult:
    """Calculate the groundstate of a given spin spiral vector q_c."""
    from gpaw import GPAW, PW

    if atoms.has('initial_magmoms'):
        magmomx = atoms.get_initial_magnetic_moments()
    else:
        magmomx = np.ones(len(atoms), float)
    magmoms = np.zeros((len(atoms), 3))
    magmoms[:, 0] = magmomx
    q_c = [eval(x) for x in filter(None, q_c.strip("[").strip("]").split(" "))]
    calc = GPAW(mode=PW(ecut, qspiral=q_c),
                xc='LDA',
                symmetry='off',
                experimental={'magmoms': magmoms, 'soc': False},
                kpts={'density': kptdensity},
                txt='gsq.txt',
                parallel={'kpt': 40})

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    totmom_v, magmom_av = calc.density.estimate_magnetic_moments()
    atoms.calc.write('gsq.gpw')
    return ASRResult.fromdata(en=energy, q=q_c, ml=magmom_av, mT=totmom_v)


def webpanel(result, context):
    from asr.database.browser import table, fig
    spiraltable = table(result, 'Property', ['bandwidth', 'minimum'],
                        context.descriptions)

    panel = {
        'title': 'Spin spirals',
        'columns': [[fig('spin_spiral_bs.png')], [spiraltable]],
        'plot_descriptions': [
            {'function': plot_bandstructure,
             'filenames': ['spin_spiral_bs.png']}
        ],
        'sort': 3}

    return [panel]


@prepare_result
class Result(ASRResult):
    path: BandPath
    energies: np.ndarray
    qmin: np.ndarray
    local_magmoms: np.ndarray
    total_magmoms: np.ndarray
    bandwidth: float
    minima: np.ndarray
    key_descriptions = {"path" : "List of Spin spiral vectors",
                        "energies" : "Potential energy [eV]",
                        "qmin" : "The q-vector with lowest energy",
                        "local_magmoms" : "List of estimated local moments [mu_B]",
                        "total_magmoms" : "Estimated total moment [mu_B]",
                        "bandwidth" : "Energy difference [meV]",
                        "minima" : "ndarray of indices of energy minima"}
    formats = {"ase_webpanel": webpanel}


@instruction(module='asr.c2db.spinspiral')
@atomsopt
@option('--q_path', help='Spin spiral high symmetry path eg. "GKMG"', type=str)
@option('--n', type=int)
def main(atoms: Atoms, q_path: str = None, n: int = 11) -> Result:
    if q_path is None:
        sp = atoms.cell.get_bravais_lattice().get_special_points()
        # Prune for special points with kz = 0
        q_path = ''
        for k, v in sp.items():
            if v[2] == 0:
                q_path += k
        q_path = q_path + q_path[0]

    path = atoms.cell.bandpath(q_path, npoints=n)
    Q = np.round(path.kpts, 16)
    energies = []
    lmagmom_av = []
    Tmagmom_v = []
    for q_c in Q:
        result = calculate(atoms, q_c=str(q_c))
        energies.append(result['en'])
        lmagmom_av.append(result['ml'])
        Tmagmom_v.append(result['mT'])

    energies = np.asarray(energies)
    lmagmom_av = np.asarray(lmagmom_av)
    Tmagmom_v = np.asarray(Tmagmom_v)

    bandwidth = (np.max(energies) - np.min(energies)) * 1000
    minima = calc_minima(energies=energies)
    qmin = Q[minima]
    return Result.fromdata(path=path, energies=energies, qmin=qmin,
                           local_magmoms=lmagmom_av, total_magmoms=Tmagmom_v,
                           bandwidth=bandwidth, minima=minima)


def calc_minima(energies):
    emin = energies[np.argmin(energies)]
    minima = np.where(np.abs(energies - emin) < 0.001)[0]
    nmin = len(minima)
    if nmin == 1:
        return [minima]

    # Note minima is a sorted list of integers
    # Check if two points are seperated by 1, then compare height of stars
    # to select the points around which interpolation is required
    # * . . *
    # * . . . *
    minRuns = streak(list(minima))
    for i, run in enumerate(minRuns):
        if len(run) == 1:
            continue

        if len(run) % 2 == 1:
            minRuns[i] = run[::2]
        else:
            if max(run) == (len(energies) - 1):
                minRuns[i] = run[0::2]
            elif min(run) == 0:
                minRuns[i] = run[1::2]
            elif energies[run[-1] + 1] <= energies[run[0] - 1]:
                minRuns[i] = run[0::2] + [run[-1] + 1]
            else:
                minRuns[i] = [run[0] - 1] + run[1::2]

    # Flatten list of lists
    minima = [item for sublist in minRuns for item in sublist]
    return minima


def streak(minima):
    # Tail recursive function
    # It seperates list of int into list into streaks of
    # incrementing-by-1 lists and single digit lists
    # [3, 5, 6, 7, 9, 10] -> [[3], [5, 6, 7], [9, 10]]
    def streak_rec(minima, previous, rest):
        if rest == []:
            return minima
        elif rest[0] - previous == 1:
            # The number is incrementing, take the last number and append this
            prev = minima.pop()
            minima.append(prev + [rest[0]])
            return streak_rec(minima, rest[0], rest[1:])
        else:
            # The number is new, just append
            minima.append([rest[0]])
            return streak_rec(minima, rest[0], rest[1:])
    return streak_rec([], -2, minima)


def streak2(minima):
    # Essentially the same for sorted arrays, positive integers
    # Not as crashes for decrementing-by-1 and negative numbers
    nmin = len(minima)
    minRuns = []
    for i in range(0, nmin):
        if minima[i] - minima[i - 1] == 1:
            prev = minRuns.pop()
            minRuns.append(prev + [minima[i]])
        else:
            minRuns.append([minima[i]])
    return minRuns


@prepare_result
class InterpolResult(ASRResult):
    path: BandPath
    energies: np.ndarray
    qmin: np.ndarray
    local_magmoms: np.ndarray
    total_magmoms: np.ndarray
    bandwidth: float
    minima: np.ndarray
    key_descriptions = {"path" : "List of Spin spiral vectors",
                        "energies" : "Potential energy [eV]",
                        "qmin" : "The q-vector with lowest energy",
                        "local_magmoms" : "List of estimated local moments [mu_B]",
                        "total_magmoms" : "Estimated total moment [mu_B]",
                        "bandwidth" : "Energy difference [meV]",
                        "minima" : "ndarray of indices of energy minima"}
    formats = {"ase_webpanel": webpanel}


@instruction(module='asr.c2db.spinspiral')
@atomsopt
@option('--path', help='Spin spiral high symmetry path eg. "GKMG"', type=str)
@option('--e_min', type=int)
def interpol_min(atoms: Atoms, path: BandPath, e_min: int = None) -> InterpolResult:
    energies = []
    lmagmom_av = []
    Tmagmom_v = []

    qL, qR, qvL, qvR = enhance(path, e_min, enh=11)
    Q = np.append(qL, qR, axis=0)
    Qv = np.append(qvL, qvR, axis=0)
    for q_c in Q:
        result = calculate(atoms, q_c=str(q_c))
        energies.append(result['en'])
        lmagmom_av.append(result['ml'])
        Tmagmom_v.append(result['mT'])

    energies = np.asarray(energies)
    lmagmom_av = np.asarray(lmagmom_av)
    Tmagmom_v = np.asarray(Tmagmom_v)

    bandwidth = (np.max(energies) - np.min(energies)) * 1000
    minidx = [e_min, np.argmin(energies)]  # [old min, new min]
    return InterpolResult.fromdata(path=[Q, Qv], energies=energies,
                                   local_magmoms=lmagmom_av, total_magmoms=Tmagmom_v,
                                   bandwidth=bandwidth, minima=minidx)


def enhance(path, e_min, enh=11):
    # Should only be called if bandwidth > noise level and/or if minimum is magnetic
    Q = path.kpts
    cell_cv = path.cell

    def interpolate(Q):
        if e_min > 0 and e_min < (len(Q) - 1):
            qs = np.take(Q, [e_min - 1, e_min, e_min + 1], axis=0)
            return interpolate_min(qs, enh)
        else:
            return interpolate_edge(Q, e_min, enh)

    qL, qR = interpolate(Q)
    from ase.dft.kpoints import kpoint_convert
    qvL = kpoint_convert(cell_cv, skpts_kc=qL) / (2 * np.pi)
    qvR = kpoint_convert(cell_cv, skpts_kc=qR) / (2 * np.pi)
    return qL, qR, qvL, qvR


def interpolate_min(qs, enh):
    lR = np.linalg.norm(qs[2] - qs[1])
    lL = np.linalg.norm(qs[1] - qs[0])

    nR = round(enh * lR / (lL + lR))
    nL = round(enh * lL / (lL + lR))

    if nL + nR != enh:
        res = enh - (nL + nR)
        if lL > lR:
            nL += res
        else:
            nR += res

    qL = np.linspace(qs[0], qs[1], nL + 2)
    qR = np.linspace(qs[1], qs[2], nR + 2)[1:]
    return qL, qR


def interpolate_edge(Q, e_min, enh):
    # Case: Both edges are equivalent, we interpolate both edges
    if (Q[0] == Q[-1]).all():
        Q = np.delete(Q, -1, axis=0)
        qs = np.take(Q, [e_min - 1, e_min, e_min + 1], axis=0, mode='wrap')
        qL, qR = interpolate_min(qs, enh)
        return qL, qR

    elif e_min == 0:
        qL = np.linspace(Q[e_min], Q[e_min + 1], enh + 2)
        qR = np.asarray([])
        return qL, qR

    elif e_min == (len(Q) - 1):
        qR = np.linspace(Q[e_min - 1], Q[e_min], enh + 2)
        qL = np.asarray([])
        return qL, qR


def plot_bandstructure(context, fname, data=None):
    from matplotlib import pyplot as plt
    if data is None:
        data = context.get_record('results-asr.spinspiral.json').result
    path = data['path']
    energies = data['energies']

    # If we want to normalize by number of magnetic atoms
    # local_magmoms =  data['local_magmoms'][0]
    # ml = np.linalg.norm(local_magmoms, axis=1)
    # nmagatoms = np.sum(ml > 0.2)
    energies = ((energies - energies[0]) * 1000)  # / nmagatoms
    q, x, X = path.get_linear_kpoint_axis()

    total_magmoms = data['total_magmoms']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Setup main energy plot
    ax1.plot(q, energies, c='C1', marker='.', label='Energy')
    ax1.set_ylim([np.min(energies * 1.1), np.max(energies * 1.15)])
    ax1.set_ylabel('Spin spiral energy [meV]')

    ax1.set_xlabel('q vector [Å$^{-1}$]')
    ax1.set_xticks(x)
    ax1.set_xticklabels([i.replace('G', r"$\Gamma$") for i in X])
    for xc in x:
        if xc != min(q) and xc != max(q):
            ax1.axvline(xc, c='gray', linestyle='--')
    ax1.margins(x=0)

    # Add spin wavelength axis
    def tick_function(X):
        lmda = 2 * np.pi / X
        return [f"{z:.1f}" for z in lmda]

    # Non-cumulative length of q-vectors to find wavelength
    Q = np.linalg.norm(2 * np.pi * path.cartesian_kpts(), axis=-1)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    idx = round(len(Q) / 5)

    ax2.set_xticks(q[::idx])
    ax2.set_xticklabels(tick_function(Q[::idx]))
    ax2.set_xlabel(r"Wave length $\lambda$ [Å]")

    # Add the magnetic moment plot
    ax3 = ax1.twinx()
    mT = abs(total_magmoms[:, 0])
    # mT = np.linalg.norm(total_magmoms, axis=-1)#mT[:, 1]#
    mT2 = abs(total_magmoms[:, 1])
    mT3 = abs(total_magmoms[:, 2])
    ax3.plot(q, mT, c='r', marker='.', label='$m_x$')
    ax3.plot(q, mT2, c='g', marker='.', label='$m_y$')
    ax3.plot(q, mT3, c='b', marker='.', label='$m_z$')

    ax3.set_ylabel(r"Total norm magnetic moment ($\mu_B$)")
    mommin = np.min(mT * 0.9)
    mommax = np.max(mT * 1.15)
    ax3.set_ylim([mommin, mommax])

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    # fig.suptitle('')
    plt.tight_layout()
    # plt.savefig(fname)
    return fig


if __name__ == '__main__':
    main.cli()
