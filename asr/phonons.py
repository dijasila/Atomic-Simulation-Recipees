from pathlib import Path

import numpy as np

from ase.parallel import world
from ase.geometry import crystal_structure_from_cell
from ase.dft.kpoints import special_paths, bandpath
from ase.io import read
from ase.phonons import Phonons

import click


@click.command()
@click.option('-n', default=2, help='Supercell size')
def phonons(n=2):
    """Calculate Phonons"""
    N = n
    from asr.utils.gpaw import GPAW
    # Remove empty files:
    if world.rank == 0:
        for f in Path().glob('phonon.*.pckl'):
            if f.stat().st_size == 0:
                f.unlink()
    world.barrier()

    params = {}
    name = 'start.json'

    # Set essential parameters for phonons
    params['symmetry'] = {'point_group': False,
                          'do_not_symmetrize_the_density': True}
    # Make sure to converge forces! Can be important
    if 'convergence' in params:
        params['convergence']['forces'] = 1e-6
    else:
        params['convergence'] = {'forces': 1e-6}

    atoms = read(name)
    fd = open('phonons-{}.txt'.format(N), 'a')
    calc = GPAW(txt=fd, **params)

    # Set initial magnetic moments
    from asr.utils import is_magnetic
    if is_magnetic():
        gsold = GPAW('gs.gpw', txt=None)
        magmoms_m = gsold.get_magnetic_moments()
        atoms.set_initial_magnetic_moments(magmoms_m)

    from asr.utils import get_dimensionality
    nd = get_dimensionality()
    if nd == 3:
        supercell = (N, N, N)
    elif nd == 2:
        supercell = (N, N, 1)
    elif nd == 1:
        supercell = (N, 1, 1)

    p = Phonons(atoms, calc, supercell=supercell)
    p.run()

    return p


def analyse(atoms, name='phonon', points=300, modes=False, q_qc=None, N=2):
    params = {}
    params['symmetry'] = {'point_group': False,
                          'do_not_symmetrize_the_density': True}

    slab = read('start.json')
    from gpaw import GPAW
    calc = GPAW(txt='phonons.txt', **params)
    from asr.utils import get_dimensionality
    nd = get_dimensionality()
    if nd == 3:
        supercell = (N, N, N)
    elif nd == 2:
        supercell = (N, N, 1)
    elif nd == 1:
        supercell = (N, 1, 1)
    p = Phonons(slab, calc, supercell=supercell)
    p.read(symmetrize=0, acoustic=False)
    cell = atoms.get_cell()
    cs = crystal_structure_from_cell(cell)
    kptpath = special_paths[cs]
    if q_qc is None:
        q_qc = bandpath(kptpath, cell, points)[0]

    out = p.band_structure(q_qc, modes=modes, born=False, verbose=False)
    if modes:
        omega_kl, u_kl = out
        return np.array(omega_kl), u_kl, q_qc
    else:
        omega_kl = out
        return np.array(omega_kl), np.array(omega_kl), q_qc


def plot_phonons(row, fname):
    import matplotlib.pyplot as plt

    freqs = row.data.get('phonon_frequencies_3d')
    if freqs is None:
        return

    gamma = freqs[0]
    fig = plt.figure(figsize=(6.4, 3.9))
    ax = fig.gca()

    x0 = -0.0005  # eV
    for x, color in [(gamma[gamma < x0], 'r'),
                     (gamma[gamma >= x0], 'b')]:
        if len(x) > 0:
            markerline, _, _ = ax.stem(x * 1000, np.ones_like(x), bottom=-1,
                                       markerfmt=color + 'o',
                                       linefmt=color + '-')
            plt.setp(markerline, alpha=0.4)
    ax.set_xlabel(r'phonon frequency at $\Gamma$ [meV]')
    ax.axis(ymin=0.0, ymax=1.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def webpanel(row, key_descriptions):
    from asr.custom import table, fig
    phonontable = table(row, 'Property',
                        ['c_11', 'c_22', 'c_12', 'bulk_modulus',
                         'minhessianeig'], key_descriptions)

    panel = ('Elastic constants and phonons',
             [[fig('phonons.png')], [phonontable]])
    things = [(plot_phonons, ['phonons.png'])]

    return panel, things


group = 'Property'
dependencies = ['asr.gs']

if __name__ == '__main__':
    phonons()
