import numpy as np
from ase.io import read, write
from ase.io.jsonio import write_json
from gpaw import GPAW, PW, LCAO
from ase.calculators.dftd3 import DFTD3
from scipy.interpolate import CubicSpline
from asr.core import command, option

def get_distance(upper, lower):
    return min(upper.positions[:, 2]) - max(lower.positions[:, 2])

def initialize(structure):
    atoms = read(structure)
    tags = atoms.get_tags()
    natoms_l1 = np.extract(tags==1, tags).shape[0]
    l1 = atoms[:natoms_l1].copy()
    l2 = atoms[natoms_l1:].copy()
    l1_z = l1.positions[:, 2]
    l2_z = l2.positions[:, 2]
    if min(l1_z) > min(l2_z):
        upper = l1.copy()
        lower = l2.copy()
    else:
        upper = l2.copy()
        lower = l1.copy()
    init_dist = get_distance(upper, lower)
    if init_dist <= 0.5:
        raise ValueError(f"Interlayer distance is too low ({init_dist} Ang)")
    else:
        return upper, lower


def calculate(upper, lower, start, stop, nsteps, calc):
    from ase.io.trajectory import TrajectoryWriter

    shifts = []
    dists = []
    energies = []
    for shift in np.linspace(start, stop, nsteps): 
        shifted = upper.copy()
        shifted.translate([0, 0, shift])
        bilayer = shifted + lower
        bilayer.calc = calc
        en = bilayer.get_potential_energy()
        dst = get_distance(shifted, lower)
        dists.append(dst)
        shifts.append(shift)
        energies.append(en)
        writer = TrajectoryWriter('zscan.traj', mode='a', atoms=bilayer)
        writer.write()
    return np.array(shifts), np.array(dists), np.array(energies)


# Get optimal shift and energy through spline interpolation
def interpolate(shifts, dists, energies, npoints):
    spline = CubicSpline(shifts, energies)
    s_grid = np.linspace(shifts[0], shifts[-1], npoints)
    d_grid = np.linspace(dists[0], dists[-1], npoints)
    energies_int = spline(s_grid)
    emin = min(energies_int)
    index_min = np.argmin(energies_int)
    shift_min = s_grid[index_min]
    dist_min = d_grid[index_min]
    return {
            "scan": {
                     "shifts": shifts,
                     "dists": dists,
                     "energies": energies
                     },
            "interpolation": {
                              "shifts": s_grid,
                              "dists": d_grid,
                              "energies": energies_int
                              },
            "emin": emin,
            "shift_min": shift_min,
            "dist_min": dist_min
            }


# Post-processing plot step
def plot(results="plot-zscan.json", mode="dist", title = None):
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
    if mode == "dist":
        xgrid_scan = scan["dists"]
        xgrid_int = interpolated["dists"]
        xmin = dct["dist_min"]
        xlabel = "Interlayer distance (Angstrom)"
    elif mode == "shifts":
        xgrid_scan = scan["shifts"]
        xgrid_int = interpolated["shifts"]
        xmin = dct["shift_min"]
        xlabel = "Applied shift (Angstrom)"
    else:
        raise ValueError("mode can only be 'dists' or 'shifts'")

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


    ax.set_title(title, fontsize = 20)
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
@option('--structure', type=str) 
@option('--start', type=float) 
@option('--stop', type=float) 
@option('--nsteps', type=int) 
@option('--npoints', type=int) 
def main(structure: str="initial.json",
         start: float=-1.0,
         stop: float=1.0,
         nsteps: int=10,
         npoints: int=700):

    print(f'nsteps = {nsteps}')
    dft = GPAW(mode='lcao',
               xc='PBE',
    	   kpts={'density': 6.0, 'gamma': True},
               occupations={'name': 'fermi-dirac',
                            'width': 0.05},
               basis='dzp')
    vdw = DFTD3(dft=dft)

    upper, lower = initialize(structure)
    shifts, distances, energies = calculate(upper, lower, start, stop, nsteps, vdw)

    results = interpolate(shifts, distances, energies, npoints)
    write_json("plot-zscan.json", results)
    
    upper.translate([0, 0, results["shift_min"]])
    newatoms = upper + lower
    write("unrelaxed.json", newatoms)


if __name__ == '__main__':
    main()
