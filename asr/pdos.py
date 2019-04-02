from collections import defaultdict
import json
import os.path as op

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import numpy as np
import ase.dft.dos
from ase import Atoms
from ase.io import jsonio
from ase.parallel import paropen
from ase.units import Ha
from ase.utils import formula_metal

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.utilities.dos import raw_orbital_LDOS, raw_spinorbit_orbital_LDOS
from _gpaw import tetrahedron_weight

from c2db import magnetic_atoms
from c2db.utils import get_spin_direction

import click


def _lti(energies, dos, kpts, M, E, W=None):
    """Faster implementation."""
    zero = energies[0]
    de = energies[1] - zero
    simplices = np.array([[0, 1, 2, 3]], np.int32)
    s = np.array([0])
    volumes = np.array([1.0])
    if W is None:
        for e in E.T:
            e = e.copy()
            m = max(0, int((e.min() - zero) / de) + 1)
            n = min(len(energies) - 1, int((e.max() - zero) / de) + 1)
            if m == n:
                continue
            for k in range(4):
                tetrahedron_weight(e, simplices, k, s, dos[m:n], energies[m:n],
                                   volumes)
    else:
        for e, w in zip(E.T, W.T):
            e = e.copy()
            w = w.copy()
            m = max(0, int((e.min() - zero) / de) + 1)
            n = min(len(energies) - 1, int((e.max() - zero) / de) + 1)
            if m == n:
                continue
            for k in range(4):
                tetrahedron_weight(e, simplices, k, s, dos[m:n], energies[m:n],
                                   volumes * w[k])


old = ase.dft.dos.ltidos


def ltidos(cell, eigs, energies, weights=None):
    x = 1 / abs(np.linalg.det(cell)) / np.prod(eigs.shape[:3]) / 6
    return old(cell, eigs, energies, weights) * x


# Monkey-patch ASE:
ase.dft.dos._lti = _lti
ase.dft.dos.ltidos = ltidos


def count(zs):
    """Count the number of occurences in a list

    Parameters:
    zs: [z1, z2, ...]-list or ndarray
        list of int's or str's

    Returns:
    out: {z1: n1, ...}
        n1 is the number of occurences of z1

    Examples:
    >>> zs = [1, 8, 8]
    >>> count(zs)
    {8: 2, 1: 1}
    >>> zs = ['H', 'O', 'O']
    >>> count(zs)
    {'O': 2, 'H': 1}
    """
    c = defaultdict(int)
    for z in zs:
        c[z] += 1
    return c


def get_l_a(zs):
    """Defines which atoms and angular momentum to project onto.

    Parameters:
    zs: [z1, z2, ...]-list or array
        list of atomic numbers (zi: int)

    Returns:
    out: {int: str, ...}-dict
        keys are atomic indices and values are a string such as 'spd'
        that determines which angular momentum to project onto or a
        given atom
    """
    zs = np.asarray(zs)
    l_a = {}
    atoms = Atoms(numbers=zs)
    mag_elements = magnetic_atoms(atoms)
    for a, (z, mag) in enumerate(zip(zs, mag_elements)):
        l_a[a] = 'spd' if mag else 'sp'
    return l_a


def dft_for_pdos(kptdens=36.0):
    from c2db.densk import nonsc
    calc = nonsc(kdens=kptdens, emptybands=20, outname='pdos')
    return calc


@click.command()
def main(calc='pdos.gpw'):
    if not op.isfile('pdos.gpw'):
        dft_for_pdos()
    dosefnosoc = dosef_nosoc()
    with paropen('dosef_nosoc.txt', 'w') as fd:
        print('{}'.format(dosefnosoc), file=fd)
    dosefsoc = dosef_soc()
    with paropen('dosef_soc.txt', 'w') as fd:
        print('{}'.format(dosefsoc), file=fd)
    pdos(calc, spinorbit=False)
    pdos(calc, spinorbit=True)


def pdos(gpwname, spinorbit=True) -> None:
    """
    Writes the projected dos to a file pdos.json or pdos_soc.json

    Parameters:
    calc: GPAW calculator object or str
        calculator with a method get_orbital_ldos
    spinorbit: bool
        spin orbit coupling
    """
    fname = 'pdos_soc.json' if spinorbit else 'pdos.json'
    if op.isfile(fname):
        return
    world = mpi.world
    calc = GPAW(gpwname, txt=None)
    if spinorbit and world.rank == 0:
        calc0 = GPAW(gpwname, communicator=mpi.serial_comm)

    zs = calc.atoms.get_atomic_numbers()
    chem_symbols = calc.atoms.get_chemical_symbols()
    efermi = calc.get_fermi_level()
    l_a = get_l_a(zs)
    kd = calc.wfs.kd

    if spinorbit:
        ldos = raw_spinorbit_orbital_LDOS
    else:
        ldos = raw_orbital_LDOS

    e = np.linspace(-10 + efermi, 10 + efermi, 2000)
    ns = calc.get_number_of_spins()
    pdos_sal = defaultdict(float)
    e_s = {}
    for spin in range(ns):
        for a in l_a:
            spec = chem_symbols[a]
            for l in l_a[a]:
                if spinorbit:
                    if world.rank == 0:
                        theta, phi = get_spin_direction()
                        energies, weights = ldos(calc0, a, spin, l, theta, phi)
                        mpi.broadcast((energies, weights))
                    else:
                        energies, weights = mpi.broadcast(None)
                else:
                    energies, weights = ldos(calc, a, spin, l)

                energies.shape = (kd.nibzkpts, -1)
                energies = energies[kd.bz2ibz_k]
                energies.shape = tuple(kd.N_c) + (-1, )
                weights.shape = (kd.nibzkpts, -1)
                weights /= kd.weight_k[:, np.newaxis]
                w = weights[kd.bz2ibz_k]
                w.shape = tuple(kd.N_c) + (-1, )
                p = ltidos(calc.atoms.cell, energies * Ha, e, w)
                key = ','.join([str(spin), str(spec), str(l)])
                pdos_sal[key] += p
        e_s[spin] = e

    data = {
        'energies': e,
        'pdos_sal': pdos_sal,
        'symbols': calc.atoms.get_chemical_symbols(),
        'efermi': efermi
    }
    with paropen(fname, 'w') as fd:
        json.dump(jsonio.encode(data), fd)


def dosef_nosoc():
    """
    Get dos at ef
    """
    name = 'pdos.gpw' if op.isfile('pdos.gpw') else 'densk.gpw'
    calc = GPAW(name, txt=None)

    from ase.dft.dos import DOS
    dos = DOS(calc, width=0.0, window=(-0.1, 0.1), npts=3)
    return dos.get_dos()[1]


def dosef_soc():
    """
    Get dos at ef
    """
    from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
    from ase.dft.dos import DOS
    from c2db.utils import gpw2eigs
    name = 'pdos.gpw' if op.isfile('pdos.gpw') else 'densk.gpw'
    world = mpi.world
    if world.rank == 0:
        calc = GPAW(name, communicator=mpi.serial_comm, txt=None)

        dos = DOS(calc, width=0.0, window=(-0.1, 0.1), npts=3)

        # hack DOS
        e_skm, ef = gpw2eigs(name, optimal_spin_direction=True)
        if e_skm.ndim == 2:
            e_skm = e_skm[np.newaxis]
        dos.nspins = 1
        dos.e_skn = e_skm - ef
        bzkpts = calc.get_bz_k_points()
        size, offset = get_monkhorst_pack_size_and_offset(bzkpts)
        bz2ibz = calc.get_bz_to_ibz_map()
        shape = (dos.nspins, ) + tuple(size) + (-1, )
        dos.e_skn = dos.e_skn[:, bz2ibz].reshape(shape)
        dos = dos.get_dos() / 2
        mpi.broadcast(dos)
    else:
        dos = mpi.broadcast(None)
    return dos[1]


def plot_pdos():
    """only for testing
    """
    efermi = GPAW('gs.gpw', txt=None).get_fermi_level()
    import matplotlib.pyplot as plt
    with paropen('pdos.json', 'r') as fd:
        data = jsonio.decode(json.load(fd))
        e = np.asarray(data['energies'])
        pdos_sal = data['pdos_sal']
        symbols = data['symbols']

    with paropen('evac.txt', 'r') as fd:
        evac = float(fd.read())

    e -= evac
    pmax = 0.0
    for s, pdos_al in pdos_sal.items():
        for a, pdos_l in sorted(pdos_al.items()):
            for l, pdos in sorted(pdos_l.items(), reverse=True):
                pdos = np.asarray(pdos)
                pmax = max(pdos.max(), pmax)
                plt.plot(pdos, e, label='{} ({})'.format(symbols[int(a)], l))
    plt.xlim(0, pmax)
    plt.ylim(efermi - evac - 2, efermi - evac + 2)
    plt.legend()
    plt.ylabel('energy relative to vacuum [eV]')
    plt.xlabel('pdos [states/eV]')
    plt.show()


def pdos_pbe(row,
             filename='pbe-pdos.png',
             figsize=(6.4, 4.8),
             fontsize=10,
             lw=2,
             loc='best'):
    if 'pdos_pbe' not in row.data:
        return

    def smooth(y, npts=3):
        return np.convolve(y, np.ones(npts) / npts, mode='same')

    dct = row.data.pdos_pbe
    e = dct['energies']
    pdos_sal2 = dct['pdos_sal']
    z_a = set(row.numbers)
    symbols = Atoms(formula_metal(z_a)).get_chemical_symbols()

    def cmp(k):
        s, a, L = k.split(',')
        si = symbols.index(k.split(',')[1])
        li = ['s', 'p', 'd', 'f'].index(L)
        return ('{}{}{}'.format(s, si, li))

    pdos_sal = {}
    for k in sorted(pdos_sal2.keys(), key=cmp):
        pdos_sal[k] = pdos_sal2[k]
    colors = {}
    i = 0
    for k in sorted(pdos_sal.keys(), key=cmp):
        if int(k[0]) == 0:
            colors[k[2:]] = 'C{}'.format(i % 10)
            i += 1
    spinpol = False
    for k in pdos_sal.keys():
        if int(k[0]) == 1:
            spinpol = True
            break
    ef = dct['efermi']
    mpl.rcParams['font.size'] = fontsize
    ax = plt.figure(figsize=figsize).add_subplot(111)
    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    emin = row.get('vbm', ef) - 3
    emax = row.get('cbm', ef) + 3
    i1, i2 = abs(e - emin).argmin(), abs(e - emax).argmin()
    pdosint_s = defaultdict(float)
    for key in sorted(pdos_sal.keys(), key=cmp):
        pdos = pdos_sal[key]
        spin, spec, lstr = key.split(',')
        spin = int(spin)
        sign = 1 if spin == 0 else -1
        pdosint_s[spin] += np.trapz(y=pdos[i1:i2], x=e[i1:i2])
        if spin == 0:
            label = '{} ({})'.format(spec, lstr)
        else:
            label = None
        ax.plot(
            smooth(pdos) * sign, e, label=label, color=colors[key[2:]], lw=lw)

    ax.legend(loc=loc)
    ax.axhline(ef, color='k', ls=':')
    ax.set_ylim(emin, emax)
    if spinpol:
        xmax = max(pdosint_s.values())
        ax.set_xlim(-xmax * 0.5, xmax * 0.5)
    else:
        ax.set_xlim(0, pdosint_s[0] * 0.5)

    xlim = ax.get_xlim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.01
    text = ax.annotate(
        r'$E_\mathrm{F}$',
        xy=(x0, ef),
        ha='left',
        va='bottom',
        fontsize=fontsize * 1.3)
    text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])
    ax.set_xlabel('projected dos [states / eV]')
    ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def webpanel(row, key_descriptions):
    panel = []
    things = [(pdos_pbe, ['pbe-pdos.png'])]

    return panel, things


group = 'Property'

if __name__ == '__main__':
    main()
