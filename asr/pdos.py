from asr.utils import option, update_defaults
import click

from collections import defaultdict
import json

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from ase import Atoms
from ase.io import jsonio
from ase.parallel import paropen
from ase.units import Ha
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset as k2so
from ase.dft.dos import DOS
from ase.dft.dos import linear_tetrahedron_integration as lti
from ase.utils.formula import formula_metal

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.utilities.dos import raw_orbital_LDOS, raw_spinorbit_orbital_LDOS


from asr.utils import magnetic_atoms
from asr.utils.gpw import gpw2eigs, get_spin_direction


def plot_pdos():
    """only for testing
    """
    from gpaw import GPAW
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


class SOCDOS():  # At some point, the GPAW DOS class should handle soc XXX
    """Hack to make DOS class work with spin orbit coupling"""
    def __init__(self, gpw, **kwargs):
        """
        Parameters:
        -----------
        gpw : str
            The SOCDOS takes a filename of the GPAW calculator object and loads
            it, instead of the normal ASE compliant calculator object.
        """
        self.gpw = gpw

        if mpi.world.rank == 0:
            self.calc = GPAW(gpw, communicator=mpi.serial_comm, txt=None)
            self.dos = DOS(self.calc, **kwargs)
        else:
            self.calc = None
            self.dos = None

    def get_dos(self):
        if mpi.world.rank == 0:  # GPAW spin-orbit correction is done in serial
            # hack dos
            e_skm, ef = gpw2eigs(self.gpw, optimal_spin_direction=True)
            if e_skm.ndim == 2:
                e_skm = e_skm[np.newaxis]
            self.dos.nspins = 1
            self.dos.e_skn = e_skm - ef
            bzkpts = self.calc.get_bz_k_points()
            size, offset = k2so(bzkpts)
            bz2ibz = self.calc.get_bz_to_ibz_map()
            shape = (self.dos.nspins, ) + tuple(size) + (-1, )
            self.dos.e_skn = self.dos.e_skn[:, bz2ibz].reshape(shape)
            dos = self.dos.get_dos() / 2
            mpi.broadcast(dos)
        else:
            dos = mpi.broadcast(None)
        return dos


def get_l_a(zs):
    """Defines which atoms and angular momentum to project onto.

    Parameters:
    -----------
    zs : [z1, z2, ...]-list or array
        list of atomic numbers (zi: int)

    Returns:
    --------
    l_a : {int: str, ...}-dict
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


def refine_gs_for_pdos(kptdens=36.0, emptybands=20):
    from asr.utils.refinegs import refinegs
    calc, gpw = refinegs(selfc=False, outf=True,
                         kptdens=kptdens, emptybands=emptybands,
                         txt='pdos.txt')
    return calc, gpw


def calculate_dos_at_ef(calc, gpw, soc=False):
    """Get dos at the Fermi energy"""
    if soc:
        dos = SOCDOS(gpw, width=0.0, window=(-0.1, 0.1), npts=3)
    else:
        dos = DOS(calc, width=0.0, window=(-0.1, 0.1), npts=3)
    return dos.get_dos()[1]


def write_dos_at_ef(dos_at_ef, soc=False):
    with paropen('dos-at-ef_soc«%s»' % str(soc), 'w') as fd:
        print('{}'.format(dos_at_ef), file=fd)


def read_dos_at_ef(soc=False):
    with paropen('dos-at-ef_soc«%s»' % str(soc), 'r') as fd:
        return float(fd.read())


def calculate_pdos(calc, gpw, soc=True):
    """Calculate the projected density of states

    Returns:
    --------
    energies : nd.array
        energies 10 eV under and above Fermi energy
    pdos_sal : defaultdict
        pdos for spin, atom and orbital angular momentum
    symbols : list
        chemical symbols in Atoms object
    efermi : float
        Fermi energy
    """
    world = mpi.world

    if soc and world.rank == 0:
        calc0 = GPAW(gpw, communicator=mpi.serial_comm, txt=None)

    zs = calc.atoms.get_atomic_numbers()
    chem_symbols = calc.atoms.get_chemical_symbols()
    efermi = calc.get_fermi_level()
    l_a = get_l_a(zs)
    kd = calc.wfs.kd

    if soc:
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
                if soc:
                    if world.rank == 0:  # GPAW soc is done in serial
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
                p = lti(calc.atoms.cell, energies * Ha, e, w)
                key = ','.join([str(spin), str(spec), str(l)])
                pdos_sal[key] += p
        e_s[spin] = e

    return e, pdos_sal, calc.atoms.get_chemical_symbols(), efermi
    

def write_pdos(energies, pdos_sal, symbols, efermi, soc=False):
    data = {'energies': energies,
            'pdos_sal': pdos_sal,
            'symbols': symbols,
            'efermi': efermi}
    with paropen('pdos_soc«%s».json' % str(soc), 'w') as fd:
        json.dump(jsonio.encode(data), fd)


def read_pdos(soc=False):
    with paropen('pdos_soc«%s».json' % str(soc), 'r') as fd:
        data = jsonio.decode(json.load(fd))
    energies = data['energies']
    pdos_sal = data['pdos_sal']
    symbols = data['symbols']
    efermi = data['efermi']
    return energies, pdos_sal, symbols, efermi


def analyse_pdos(energies, pdos_sal, symbols, efermi):
    """
    Subtract the vacuum energy.
    """
    # get evac XXX
    evac = 0.

    e = energies - evac
    ef = efermi - evac

    # maybe use "old_to_new" pdos_sal? XXX

    return {'pdos_sal': pdos_sal, 'energies': e, 'efermi': ef}


def get_results():
    """All distributable results calculated by pdos.py"""
    return {'dos_at_ef_nosoc': read_dos_at_ef(soc=False),
            'dos_at_ef_soc': read_dos_at_ef(soc=True),
            'pdos_data_nosoc': analyse_pdos(*read_pdos(soc=False)),
            'pdos_data_soc': analyse_pdos(*read_pdos(soc=True))}


def write_results(results):
    with paropen('results_pdos.json', 'w') as fd:
        json.dump(jsonio.encode(results), fd)


def read_results():
    with paropen('results_pdos.json', 'r') as fd:
        results = jsonio.decode(json.load(fd))
    return results


@click.command()
@update_defaults('asr.pdos')
@option('--kptdens', default=36.0,
        help='k-point density')
@option('--emptybands', default=20,
        help='number of empty bands to include')
def main(kptdens, emptybands):
    # Refine ground state with more k-points
    calc, gpw = refine_gs_for_pdos(kptdens, emptybands)

    # Calculate and write the dos at the Fermi energy
    write_dos_at_ef(calculate_dos_at_ef(calc, gpw, soc=False), soc=False)
    write_dos_at_ef(calculate_dos_at_ef(calc, gpw, soc=True), soc=True)

    # Calculate and write the pdos
    write_pdos(*calculate_pdos(calc, gpw, soc=False), soc=False)
    write_pdos(*calculate_pdos(calc, gpw, soc=True), soc=True)

    write_results(get_results())


def collect_data(atoms):
    kvp = {}
    data = {}
    key_descriptions = {}  # what does key_descriptions refer to? XXX

    results = read_results()

    kvp['dos_at_ef_nosoc'] = results['dos_at_ef_nosoc']
    kvp['dos_at_ef_soc'] = results['dos_at_ef_soc']
    data['pdos_nosoc'] = results['pdos_data_nosoc']
    data['pdos_soc'] = results['pdos_data_soc']
    
    return kvp, key_descriptions, data


def webpanel(row, key_descriptions):
    panel = []
    things = [(pdos_pbe, ['pbe-pdos.png'])]

    return panel, things


group = 'Property'
resources = '8:1h'  # How many resources are used
dependencies = ['asr.gs']

if __name__ == '__main__':
    main()
