import numpy as np
from ase.io import read, write
from ase.io.jsonio import write_json
from gpaw import GPAW, PW
from ase.calculators.dftd3 import DFTD3
from scipy.interpolate import CubicSpline
from asr.core import command, option
from asr.utils.moireutils import Bilayer


def calculate(atoms, rng, calc):
    from ase.io.trajectory import TrajectoryWriter
    dists = []
    energies = []
    for dist in rng:
        atoms.set_interlayer_distance(dist)
        atoms.center()
        atoms.calc = calc
        en = atoms.get_potential_energy()
        dst = atoms.get_interlayer_distance()
        dists.append(dst)
        energies.append(en)
        writer = TrajectoryWriter('zscan.traj', mode='a', atoms=atoms)
        writer.write()
    return np.array(dists), np.array(energies)


def interpolate(dists, energies, npoints):
    '''Get optimal shift and energy through spline interpolation'''
    spline = CubicSpline(dists, energies)
    d_grid = np.linspace(dists[0], dists[-1], npoints)
    energies_int = spline(d_grid)
    emin = min(energies_int)
    index_min = np.argmin(energies_int)
    dist_min = d_grid[index_min]
    return {
        "scan": {
            "dists": dists,
            "energies": energies
        },
        "interpolation": {
            "dists": d_grid,
            "energies": energies_int
        },
        "dist_min": dist_min,
        "emin": emin
    }


# Post-processing plot step
@command('asr.zscan@plot')
@option('--results', type=str)
@option('--title', type=str)
def plot(results="plot-zscan.json", title=None):
    import matplotlib.pyplot as plt
    from ase.io.jsonio import read_json

    dct = read_json(results)
    ax = plt.figure(figsize=(12, 9)).add_subplot(111)

    style_int = dict(
        color='#8c8989',
        ls='--',
        lw=2.5,
        zorder=0)
    style_scan = dict(
        color='#546caf',
        marker='o',
        s=40,
        zorder=1)
    style_min = dict(
        color='C1',
        marker='o',
        s=50,
        zorder=2)

    scan = dct["scan"]
    interpolated = dct["interpolation"]
    xgrid_scan = scan["dists"]
    xgrid_int = interpolated["dists"]
    xmin = dct["dist_min"]
    xlabel = "Interlayer distance (Angstrom)"

    ax.scatter(
        xgrid_scan,
        scan["energies"],
        **style_scan,
        label='Scan')
    ax.plot(
        xgrid_int,
        interpolated["energies"],
        **style_int,
        label='Interpolation')
    ax.scatter(
        xmin,
        dct["emin"],
        **style_min,
        label='Optimal value')

    ax.set_title(title, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("Energy (eV)", fontsize=20)
    ax.xaxis.set_tick_params(width=3, length=10)
    ax.yaxis.set_tick_params(width=3, length=10)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.legend(fontsize=20)
    plt.show()


@command('asr.zscan')
@option('--structure', type=str, help='Starting structure file')
@option('--start', type=float, help='Starting shift')
@option('--stop', type=float, help='Final shift')
@option('--nsteps', type=int, help='Number of energy evaluations')
@option('--npoints', type=int, help='Number of points for interpolation')
def main(structure: str = "initial.json",
         start: float = 3.0,
         stop: float = 4.0,
         nsteps: int = 10,
         npoints: int = 700):

    dft = GPAW(mode='lcao',
               xc='PBE',
               kpts={'density': 6.0, 'gamma': True},
               occupations={'name': 'fermi-dirac',
                            'width': 0.05},
               basis='dzp')
    vdw = DFTD3(dft=dft)

    atoms = Bilayer(read(structure))

    rng = np.linspace(start, stop, nsteps)
    distances, energies = calculate(atoms, rng, vdw)

    results = interpolate(distances, energies, npoints)
    write_json("plot-zscan.json", results)
    atoms.set_interlayer_distance(results["dist_min"])
    atoms.center()
    write("unrelaxed.json", atoms)


if __name__ == '__main__':
    main.cli()
