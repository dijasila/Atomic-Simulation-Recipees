from pathlib import Path

import numpy as np

from ase.parallel import world
from ase.io import read
from ase.phonons import Phonons

from asr.core import command, option


def creates():
    atoms = read('structure.json')
    natoms = len(atoms)
    filenames = ['phonon.eq.pckl']
    for a in range(natoms):
        for v in 'xyz':
            for pm in '+-':
                # Atomic forces for a displacement of atom a in direction v
                filenames.append(f'phonon.{a}{v}{pm}.pckl')
    return filenames


def todict(filename):
    from ase.utils import pickleload
    return {'contents': pickleload(open(filename, 'rb')),
            'write': 'asr.phonons@tofile'}


def tofile(filename, contents):
    from ase.utils import opencew
    import pickle
    fd = opencew(filename)
    if world.rank == 0:
        pickle.dump(contents, fd, protocol=2)
        fd.close()


@command('asr.phonons',
         requires=['structure.json', 'gs.gpw'],
         dependencies=['asr.gs'],
         creates=creates,
         todict={'.pckl': todict})
@option('-n', help='Supercell size')
@option('--ecut', help='Energy cutoff')
@option('--kptdensity', help='Kpoint density')
@option('--fconverge', help='Force convergence criterium')
def calculate(n=2, ecut=800, kptdensity=6.0, fconverge=1e-4):
    """Calculate atomic forces used for phonon spectrum."""
    from asr.calculators import get_calculator
    # Remove empty files:
    if world.rank == 0:
        for f in Path().glob('phonon.*.pckl'):
            if f.stat().st_size == 0:
                f.unlink()
    world.barrier()

    params = {'mode': {'name': 'pw', 'ecut': ecut},
              'kpts': {'density': kptdensity, 'gamma': True}}

    # Set essential parameters for phonons
    params['symmetry'] = {'point_group': False}
    # Make sure to converge forces! Can be important
    params['convergence'] = {'forces': fconverge}

    atoms = read('structure.json')
    fd = open('phonons.txt'.format(n), 'a')
    calc = get_calculator()(txt=fd, **params)

    # Set initial magnetic moments
    from asr.core import is_magnetic
    if is_magnetic():
        gsold = get_calculator()('gs.gpw', txt=None)
        magmoms_m = gsold.get_magnetic_moments()
        atoms.set_initial_magnetic_moments(magmoms_m)

    from asr.core import get_dimensionality
    nd = get_dimensionality()
    if nd == 3:
        supercell = (n, n, n)
    elif nd == 2:
        supercell = (n, n, 1)
    elif nd == 1:
        supercell = (n, 1, 1)

    p = Phonons(atoms=atoms, calc=calc, supercell=supercell)
    p.run()


def requires():
    return creates() + ['results-asr.phonons@calculate.json']


def webpanel(row, key_descriptions):
    from asr.browser import table, fig
    phonontable = table(row, 'Property',
                        ['c_11', 'c_22', 'c_12', 'bulk_modulus',
                         'minhessianeig'], key_descriptions)

    panel = {'title': 'Elastic constants and phonons',
             'columns': [[fig('phonons.png')], [phonontable]],
             'plot_descriptions': [{'function': plot_phonons,
                                    'filenames': ['phonons.png']}]}

    return [panel]


@command('asr.phonons',
         requires=requires,
         webpanel=webpanel,
         dependencies=['asr.phonons@calculate'])
def main():
    from asr.core import read_json
    from asr.core import get_dimensionality
    dct = read_json('results-asr.phonons@calculate.json')
    atoms = read('structure.json')
    n = dct['__params__']['n']
    nd = get_dimensionality()
    if nd == 3:
        supercell = (n, n, n)
    elif nd == 2:
        supercell = (n, n, 1)
    elif nd == 1:
        supercell = (n, 1, 1)
    p = Phonons(atoms=atoms, supercell=supercell)
    p.read()
    q_qc = np.indices(p.N_c).reshape(3, -1).T / p.N_c
    out = p.band_structure(q_qc, modes=True, born=False, verbose=False)
    omega_kl, u_kl = out
    results = {'omega_kl': omega_kl,
               'u_kl': u_kl}

    return results


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


if __name__ == '__main__':
    main.cli()
