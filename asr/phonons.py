from pathlib import Path
import numpy as np
from ase.parallel import world
from ase.io import read
from ase.phonons import Phonons
from asr.utils import command, option


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
         dependencies=['asr.gs@calculate'],
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
    from asr.utils import is_magnetic
    if is_magnetic():
        gsold = get_calculator()('gs.gpw', txt=None)
        magmoms_m = gsold.get_magnetic_moments()
        atoms.set_initial_magnetic_moments(magmoms_m)

    from asr.utils import get_dimensionality
    nd = get_dimensionality()
    if nd == 3:
        supercell = (n, n, n)
    elif nd == 2:
        supercell = (n, n, 1)
    elif nd == 1:
        supercell = (1, 1, n)

    p = Phonons(atoms=atoms, calc=calc, supercell=supercell)
    p.run()


def requires():
    return creates() + ['results-asr.phonons@calculate.json']


def webpanel(row, key_descriptions):
    from asr.utils.custom import table, fig
    print('in phonons webpanel')
    phonontable = table(row, 'Property',
                        ['c_11', 'c_22', 'c_12', 'bulk_modulus',
                         'minhessianeig'], key_descriptions)

    panel = {'title': 'Elastic constants and phonons',
             'columns': [[fig('phonons.png')], [phonontable]],
             'plot_descriptions': [{'function': plot,
                                    'filenames': ['phonons.png']}]}

    return [panel]


@command('asr.phonons',
         requires=requires,
         webpanel=webpanel,
         dependencies=['asr.phonons@calculate'])
def main():
    from asr.utils import read_json
    from asr.utils import get_dimensionality
    dct = read_json('results-asr.phonons@calculate.json')
    atoms = read('structure.json')
    n = dct['__params__']['n']
    nd = get_dimensionality()
    if nd == 3:
        supercell = (n, n, n)
    elif nd == 2:
        supercell = (n, n, 1)
    elif nd == 1:
        supercell = (1, 1, n)
    p = Phonons(atoms=atoms, supercell=supercell)
    p.read()
    q_qc = np.indices(p.N_c).reshape(3, -1).T / p.N_c
    out = p.band_structure(q_qc, modes=True, born=False, verbose=False)
    omega_kl, u_klav = out
    results = {'omega_kl': omega_kl,
               'u_kl': u_kl}

    return results


def plot(row, filename):
    import matplotlib.pyplot as plt
    print('in phonon plot')


    gamma = []
    atoms = row.toatoms()
    data=row.data['results-asr.phonons.json']
    omega_kl = data['omega_kl']
    u_kl = data['u_kl']

    for l in range(3*len(atoms)):
         print('mode:', l+1)
         modsum = np.linalg.norm(u_kl[0,l].sum(axis=-2))
         summod = np.linalg.norm(u_kl[0,l],axis=-1).sum()
         print(omega_kl[0,l], modsum/summod)
         if abs(modsum/summod-1) >= 0.01:
             gamma.append(omega_kl[0,l])

    gamma = np.array(gamma)
    fig = plt.figure(figsize=(6.4, 3.9))
    ax = fig.gca()
    gamma = np.array(gamma)

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
    #print('saving to file', filename)
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    main.cli()
