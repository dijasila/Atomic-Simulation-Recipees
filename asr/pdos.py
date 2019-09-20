from asr.core import command, subresult, option

from collections import defaultdict

import numpy as np

from ase import Atoms
from ase.units import Ha
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset as k2so
from ase.dft.dos import DOS
from ase.dft.dos import linear_tetrahedron_integration as lti

from asr.core import magnetic_atoms


# ---------- GPAW hacks ---------- #


class SOCDOS(DOS):
    """Hack to make DOS class work with spin orbit coupling"""
    def __init__(self, gpw, **kwargs):
        """
        Parameters:
        -----------
        gpw : str
            The SOCDOS takes a filename of the GPAW calculator object and loads
            it, instead of the normal ASE compliant calculator object.
        """
        # Initiate DOS with serial communicator instead
        from gpaw import GPAW
        import gpaw.mpi as mpi
        from asr.utils.gpw2eigs import gpw2eigs
        calc = GPAW(gpw, communicator=mpi.serial_comm, txt=None)
        DOS.__init__(self, calc, **kwargs)

        # Hack the number of spins
        self.nspins = 1

        # Hack the eigenvalues
        e_skm, ef = gpw2eigs(gpw, optimal_spin_direction=True)
        if e_skm.ndim == 2:
            e_skm = e_skm[np.newaxis]
        e_skn = e_skm - ef
        bzkpts = calc.get_bz_k_points()
        size, offset = k2so(bzkpts)
        bz2ibz = calc.get_bz_to_ibz_map()
        shape = (self.nspins, ) + tuple(size) + (-1, )
        self.e_skn = e_skn[:, bz2ibz].reshape(shape)

    def get_dos(self):
        return DOS.get_dos(self) / 2


# ---------- Main functionality ---------- #


@command('asr.pdos')
@option('--kptdensity', help='k-point density')
@option('--emptybands', help='number of empty bands to include')
def main(kptdensity=36.0, emptybands=20):  # subresults need params for log
    params = dict(kptdensity=kptdensity,
                  emptybands=emptybands)
    # Refine ground state with more k-points
    calc, gpw = refine_gs_for_pdos(kptdensity, emptybands)

    results = {}

    # ----- Slow steps ----- #
    # Calculate pdos (stored in tmpresults_pdos.json until recipe is completed)
    results['pdos_nosoc'] = pdos_nosoc(params, calc, gpw)  # subresults need
    results['pdos_soc'] = pdos_soc(params, calc, gpw)  # context to log params

    # ----- Fast steps ----- #
    # Calculate the dos at the Fermi energy
    results['dos_at_ef_nosoc'] = calculate_dos_at_ef(calc, gpw, soc=False)
    results['dos_at_ef_soc'] = calculate_dos_at_ef(calc, gpw, soc=True)

    return results


# ---------- Recipe methodology ---------- #


def refine_gs_for_pdos(kptdensity=36.0, emptybands=20):
    from asr.utils.refinegs import refinegs
    calc, gpw = refinegs(selfc=False,
                         kptdensity=kptdensity, emptybands=emptybands,
                         gpw='default', txt='default')
    return calc, gpw


# ----- PDOS ----- #

@subresult('asr.pdos')
def pdos_nosoc(calc, gpw):
    return pdos(calc, gpw, soc=False)


@subresult('asr.pdos')
def pdos_soc(calc, gpw):
    return pdos(calc, gpw, soc=True)


def pdos(calc, gpw, soc=True):
    """Main functionality to do a single pdos calculation"""
    # Do calculation
    e_e, pdos_syl, symbols, ef = calculate_pdos(calc, gpw, soc=soc)

    # Subtract the vacuum energy
    from asr.gs import get_evac
    evac = get_evac()
    if evac is not None:
        e_e -= evac
        ef -= evac

    subresults = {'pdos_syl': pdos_syl, 'symbols': symbols,
                  'energies': e_e, 'efermi': ef}

    return subresults


def calculate_pdos(calc, gpw, soc=True):
    """Calculate the projected density of states

    Returns:
    --------
    energies : nd.array
        energies 10 eV under and above Fermi energy
    pdos_syl : defaultdict
        pdos for spin, symbol and orbital angular momentum
    symbols : list
        chemical symbols in Atoms object
    efermi : float
        Fermi energy
    """
    from gpaw import GPAW
    import gpaw.mpi as mpi
    from gpaw.utilities.dos import raw_orbital_LDOS, raw_spinorbit_orbital_LDOS
    from asr.utils.gpw2eigs import get_spin_direction
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

    ns = calc.get_number_of_spins()
    theta, phi = get_spin_direction()
    # We want to extract the pdos +-10 eV from efermi
    e_e = np.linspace(-10 + efermi, 10 + efermi, 2000)
    # We distinguish in (spin(s), chemical symbol(y), angular momentum (l)),
    # that is if there are multiple atoms in the unit cell of the same chemical
    # species, their pdos are added together.
    pdos_syl = defaultdict(float)
    for spin in range(ns):
        for a in l_a:
            symbol = chem_symbols[a]
            for l in l_a[a]:
                if soc:
                    if world.rank == 0:  # GPAW soc is done in serial
                        energies, weights = ldos(calc0, a, spin, l, theta, phi)
                        mpi.broadcast((energies, weights))
                    else:
                        energies, weights = mpi.broadcast(None)
                else:
                    energies, weights = ldos(calc, a, spin, l)

                # Reshape energies
                energies.shape = (kd.nibzkpts, -1)
                energies = energies[kd.bz2ibz_k]
                energies.shape = tuple(kd.N_c) + (-1, )

                # Get true weights and reshape
                weights.shape = (kd.nibzkpts, -1)
                weights /= kd.weight_k[:, np.newaxis]
                w = weights[kd.bz2ibz_k]
                w.shape = tuple(kd.N_c) + (-1, )

                # Linear tetrahedron integration
                p = lti(calc.atoms.cell, energies * Ha, e_e, w)

                # Store in dictionary
                key = ','.join([str(spin), str(symbol), str(l)])
                pdos_syl[key] += p

    return e_e, pdos_syl, calc.atoms.get_chemical_symbols(), efermi


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
    lantha = range(58, 72)
    acti = range(90, 104)

    zs = np.asarray(zs)
    l_a = {}
    atoms = Atoms(numbers=zs)
    mag_elements = magnetic_atoms(atoms)
    for a, (z, mag) in enumerate(zip(zs, mag_elements)):
        if z in lantha or z in acti:
            l_a[a] = 'spdf'
        else:
            l_a[a] = 'spd' if mag else 'sp'
    return l_a


# ----- DOS at Fermi energy ----- #


def calculate_dos_at_ef(calc, gpw, soc=True):
    """Get dos at the Fermi energy"""
    if soc:
        dos = SOCDOS(gpw, width=0.0, window=(-0.1, 0.1), npts=3)
    else:
        dos = DOS(calc, width=0.0, window=(-0.1, 0.1), npts=3)
    return dos.get_dos()[1]


# ---------- Database and webpanel ---------- #


'''  # New format
def collect_data(results):
    kd = {}

    kd['pdos_nosoc'] = 'Projected density of states '\
        + 'without spin-orbit coupling '\
        + '(PDOS without soc)'

    kd['pdos_soc'] = 'Projected density of states '\
        + 'with spin-orbit coupling '\
        + '(PDOS with soc)'

    kd['dos_at_ef_nosoc'] = 'Density of states at the Fermi energy '\
        + 'without spin-orbit coupling '\
        + '(DOS at ef without soc) [states/eV] (KVP)'

    kd['dos_at_ef_soc'] = 'Density of states at the Fermi energy '\
        + 'with spin-orbit coupling '\
        + '(DOS at ef with soc) [states/eV] (KVP)'

    results.update({'__key_descriptions__': kd})

    return results
'''


# Old format
def collect_data(atoms):
    kd = {}

    kd['pdos_nosoc'] = ('PDOS without soc',
                        'Projected density of states '
                        + 'without spin-orbit coupling',
                        '')

    kd['pdos_soc'] = ('PDOS with soc',
                      'Projected density of states '
                      + 'with spin-orbit coupling',
                      '')

    kd['dos_at_ef_nosoc'] = ('DOS at ef without soc',
                             'Density of states at the Fermi energy '
                             + 'without spin-orbit coupling',
                             'states/eV')

    kd['dos_at_ef_soc'] = ('DOS at ef with soc',
                           'Density of states at the Fermi energy '
                           + 'with spin-orbit coupling ',
                           'states/eV')

    from asr.core import read_json
    results = read_json('results_pdos.json')
    kvp = {'dos_at_ef_nosoc': results['dos_at_ef_nosoc'],
           'dos_at_ef_soc': results['dos_at_ef_soc']}
    data = {'pdos_nosoc': results['pdos_nosoc'],
            'pdos_soc': results['pdos_soc']}

    return kvp, kd, data


def webpanel(row, key_descriptions):
    # PDOS plot goes to Electronic band structure (PBE) panel, which is
    # defined in the bandstructure recipe
    panel = ()
    things = [(plot_pdos, ['pbe-pdos.png'])]
    return panel, things


# ---------- Plotting ---------- #


def get_ordered_syl_dict(dct_syl, symbols):
    """Order a dictionary with syl keys

    Parameters
    ----------
    dct_syl : dict
        Dictionary with keys f'{s},{y},{l}'
        (spin (s), chemical symbol (y), angular momentum (l))
    symbols : list
        Sort symbols after index in this list

    Returns
    -------
    outdct_syl : OrderedDict
        Sorted dct_syl
    """
    from collections import OrderedDict

    # Setup ssili (spin, symbol index, angular momentum index) key
    def ssili(syl):
        s, a, L = syl.split(',')
        # Symbols list can have multiple entries of the same symbol
        # ex. ['O', 'Fe', 'O']. In this case 'O' will have index 0 and
        # 'Fe' will have index 1.
        si = symbols.index(a)
        li = ['s', 'p', 'd', 'f'].index(L)
        return f'{s}{si}{li}'

    return OrderedDict(sorted(dct_syl.items(), key=lambda t: ssili(t[0])))


def get_yl_colors(dct_syl):
    """Get the color indices corresponding for each symbol and angular momentum
    
    Parameters
    ----------
    dct_syl : OrderedDict
        Ordered dictionary with keys f'{s},{y},{l}'
        (spin (s), chemical symbol (y), angular momentum (l))

    Returns
    -------
    color_yl : OrderedDict
        Color strings for each symbol and angular momentum
    """
    from collections import OrderedDict

    color_yl = OrderedDict()
    c = 0
    for key in dct_syl:
        # Do not differentiate spin by color
        if int(key[0]) == 0:  # if spin is 0
            color_yl[key[2:]] = 'C{}'.format(c)
            c += 1
            c = c % 10  # only 10 colors available in cycler

    return color_yl


def plot_pdos(row, filename, soc=True,
              figsize=(6.4, 4.8), fontsize=10, lw=2, loc='best'):

    def smooth(y, npts=3):
        return np.convolve(y, np.ones(npts) / npts, mode='same')

    # Check if pdos data is stored in row
    pdos = 'pdos_soc' if soc else 'pdos_nosoc'
    if pdos not in row.data:
        return

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    # Extract raw data
    data = row.data[pdos]
    symbols = data['symbols']
    pdos_syl = get_ordered_syl_dict(data['pdos_syl'], symbols)
    e_e = data['energies']
    ef = data['efermi']

    color_yl = get_yl_colors(pdos_syl)

    # Figure out if pdos has been calculated for more than one spin channel
    spinpol = False
    for k in pdos_syl.keys():
        if int(k[0]) == 1:
            spinpol = True
            break

    # Set up plot
    mpl.rcParams['font.size'] = fontsize
    ax = plt.figure(figsize=figsize).add_subplot(111)
    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())

    # Set up energy range to plot in
    emin = row.get('vbm', ef) - 3
    emax = row.get('cbm', ef) + 3
    i1, i2 = abs(e_e - emin).argmin(), abs(e_e - emax).argmin()

    # Plot pdos
    pdosint_s = defaultdict(float)
    for key in pdos_syl:
        pdos = pdos_syl[key]
        spin, symbol, lstr = key.split(',')
        spin = int(spin)
        sign = 1 if spin == 0 else -1

        # Integrate pdos to find suiting pdos range
        pdosint_s[spin] += np.trapz(y=pdos[i1:i2], x=e_e[i1:i2])

        # Label atomic symbol and angular momentum
        if spin == 0:
            label = '{} ({})'.format(symbol, lstr)
        else:
            label = None

        ax.plot(smooth(pdos) * sign, e_e,
                label=label, color=color_yl[key[2:]], lw=lw)

    ax.legend(loc=loc)
    ax.axhline(ef, color='k', ls=':')

    # Set up axis limits
    ax.set_ylim(emin, emax)
    if spinpol:  # Use symmetric limits
        xmax = max(pdosint_s.values())
        ax.set_xlim(-xmax * 0.5, xmax * 0.5)
    else:
        ax.set_xlim(0, pdosint_s[0] * 0.5)

    # Annotate E_F
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
    if row.get('evac') is not None:
        ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
    else:
        ax.set_ylabel(r'$E$ [eV]')

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# ---------- ASR globals and main ---------- #


group = 'property'
resources = '8:1h'  # How many resources are used? XXX
dependencies = ['asr.structureinfo', 'asr.gs']

tests = []
tests.append({'description': 'Test the pdos of Si (cores=1)',
              'name': 'test_asr.pdos_Si_serial',
              'cli': ['asr run setup.materials -s Si2',
                      'ase convert materials.json structure.json',
                      'asr run setup.params '
                      'asr.gs:ecut 200 asr.gs:kptdensity 2.0 '
                      'asr.pdos:kptdensity 3.0 asr.pdos:emptybands 5',
                      'asr run pdos',
                      'asr run database.fromtree',
                      'asr run browser --only-figures']})
tests.append({'description': 'Test the pdos of Si (cores=2)',
              'name': 'test_asr.pdos_Si_parallel',
              'cli': ['asr run setup.materials -s Si2',
                      'ase convert materials.json structure.json',
                      'asr run setup.params '
                      'asr.gs:ecut 200 asr.gs:kptdensity 2.0 '
                      'asr.pdos:kptdensity 3.0 asr.pdos:emptybands 5',
                      'asr run -p 2 pdos',
                      'asr run database.fromtree',
                      'asr run browser --only-figures']})

if __name__ == '__main__':
    main.cli()
