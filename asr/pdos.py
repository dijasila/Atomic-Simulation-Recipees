from asr.core import command, subresult, option

from collections import defaultdict

import numpy as np

from ase import Atoms
from ase.units import Ha
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset as k2so
from ase.dft.dos import DOS
from ase.dft.dos import linear_tetrahedron_integration as lti

from ase.units import Hartree
from gpaw.utilities.dos import get_angular_projectors
from gpaw.spinorbit import get_spinorbit_eigenvalues

from asr.core import magnetic_atoms


# ---------- GPAW hacks ---------- #


# Hack the density of states
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


# Hack the local density of states to keep spin-orbit results and not
# compute them repeatedly
class SOCDescriptor:
    """Descriptor for spin-orbit corrections.
    [Developed and tested for raw_spinorbit_orbital_LDOS only]
    """

    def __init__(self, paw):
        self.paw = paw

        # Log eigenvalues and wavefunctions for multiple spin directions
        self.theta_d = []
        self.phi_d = []
        self.eps_dmk = []
        self.v_dknm = []

    def calculate_soc_eig(self, theta, phi):
        eps_mk, v_knm = get_spinorbit_eigenvalues(self.paw, return_wfs=True,
                                                  theta=theta, phi=phi)
        self.theta_d.append(theta)
        self.phi_d.append(phi)
        self.eps_dmk.append(eps_mk)
        self.v_dknm.append(v_knm)

    def get_soc_eig(self, theta=0, phi=0):
        # Check if eigenvalues have been computed already
        for d, (t, p) in enumerate(zip(self.theta_d, self.phi_d)):
            if abs(t - theta) < np.pi * 1.e-4 and abs(p - phi) < np.pi * 1.e-4:
                return self.eps_dmk[d], self.v_dknm[d]
        # Calculate if not
        self.calculate_soc_eig(theta, phi)
        return self.eps_dmk[-1], self.v_dknm[-1]


def raw_spinorbit_orbital_LDOS_hack(paw, a, spin, angular='spdf',
                                    theta=0, phi=0):
    """Hack raw_spinorbit_orbital_LDOS"""

    from gpaw.spinorbit import get_spinorbit_projections

    # Attach SOCDescriptor to the calculator object
    if not hasattr(paw, 'socd'):
        paw.socd = SOCDescriptor(paw)

    # Get eigenvalues and wavefunctions from SOCDescriptor
    eps_mk, v_knm = paw.socd.get_soc_eig(theta, phi)
    e_mk = eps_mk / Hartree

    # Do the rest as usual:
    ns = paw.wfs.nspins
    w_k = paw.wfs.kd.weight_k
    nk = len(w_k)
    nb = len(e_mk)

    if a < 0:
        # Allow list-style negative indices; we'll need the positive a for the
        # dictionary lookup later
        a = len(paw.wfs.setups) + a

    setup = paw.wfs.setups[a]
    energies = np.empty(nb * nk)
    weights_xi = np.empty((nb * nk, setup.ni))
    x = 0
    for k, w in enumerate(w_k):
        energies[x:x + nb] = e_mk[:, k]
        P_ami = get_spinorbit_projections(paw, k, v_knm[k])
        if ns == 2:
            weights_xi[x:x + nb, :] = w * np.absolute(P_ami[a][:, spin::2])**2
        else:
            weights_xi[x:x + nb, :] = w * np.absolute(P_ami[a][:, 0::2])**2 / 2
            weights_xi[x:x + nb, :] += w * np.absolute(P_ami[a][:, 1::2])**2  / 2
        x += nb

    if angular is None:
        return energies, weights_xi
    elif isinstance(angular, int):
        return energies, weights_xi[:, angular]
    else:
        projectors = get_angular_projectors(setup, angular, type='bound')
        weights = np.sum(np.take(weights_xi,
                                 indices=projectors, axis=1), axis=1)
        return energies, weights


# ---------- Recipe tests ---------- #

ctests = []
ctests.append({'description': 'Test the refined ground state of Si',
               'name': 'test_asr.pdos_Si_gpw',
               'cli': ['asr run "setup.materials -s Si2"',
                       'ase convert materials.json structure.json',
                       'asr run "setup.params '
                       'asr.gs@calculate:ecut 200 '
                       'asr.gs@calculate:kptdensity 2.0 '
                       'asr.pdos@calculate:kptdensity 3.0 '
                       'asr.pdos@calculate:emptybands 5"',
                       'asr run gs',
                       'asr run pdos@calculate',
                       'asr run database.fromtree',
                       'asr run "browser --only-figures"']})

tests = []
tests.append({'description': 'Test the pdos of Si (cores=1)',
              'name': 'test_asr.pdos_Si_serial',
              'cli': ['asr run "setup.materials -s Si2"',
                      'ase convert materials.json structure.json',
                      'asr run "setup.params '
                      'asr.gs@calculate:ecut 200 '
                      'asr.gs@calculate:kptdensity 2.0 '
                      'asr.pdos@calculate:kptdensity 3.0 '
                      'asr.pdos@calculate:emptybands 5"',
                      'asr run gs',
                      'asr run pdos',
                      'asr run database.fromtree',
                      'asr run "browser --only-figures"']})
tests.append({'description': 'Test the pdos of Si (cores=2)',
              'name': 'test_asr.pdos_Si_parallel',
              'cli': ['asr run "setup.materials -s Si2"',
                      'ase convert materials.json structure.json',
                      'asr run "setup.params '
                      'asr.gs@calculate:ecut 200 '
                      'asr.gs@calculate:kptdensity 2.0 '
                      'asr.pdos@calculate:kptdensity 3.0 '
                      'asr.pdos@calculate:emptybands 5"',
                      'asr run gs',
                      'asr run -p 2 pdos',
                      'asr run database.fromtree',
                      'asr run "browser --only-figures"']})


# ---------- Webpanel ---------- #


def webpanel(row, key_descriptions):
    # PDOS plot goes to Electronic band structure (PBE) panel, which is
    # defined in the bandstructure recipe

    panel = {'plot_descriptions': [{'function': plot_pdos,
                                    'filenames': ['pbe-pdos.png']},
                                   ]
             }
    return [panel]


# ---------- Main functionality ---------- #


# ----- Slow steps ----- #


@command(module='asr.pdos',
         creates=['pdos.gpw'],
         tests=ctests,
         requires=['gs.gpw'],
         dependencies=['asr.gs'])
@option('-k', '--kptdensity', type=float, help='K-point density')
@option('--emptybands', type=int, help='number of empty bands to include')
def calculate(kptdensity=20.0, emptybands=20):
    from asr.utils.refinegs import refinegs
    refinegs(selfc=False,
             kptdensity=kptdensity, emptybands=emptybands,
             gpw='pdos.gpw', txt='pdos.txt')


# ----- Fast steps ----- #


@command(module='asr.pdos',
         requires=['results-asr.gs.json', 'pdos.gpw'],
         tests=tests,
         dependencies=['asr.gs', 'asr.pdos@calculate'],
         webpanel=webpanel)
def main():
    from gpaw import GPAW

    # Get refined ground state with more k-points
    calc = GPAW('pdos.gpw', txt=None)

    results = {}

    # Calculate pdos
    results['pdos_nosoc'] = pdos(calc, 'pdos.gpw', soc=False)
    results['pdos_soc'] = pdos(calc, 'pdos.gpw', soc=True)

    # Calculate the dos at the Fermi energy
    results['dos_at_ef_nosoc'] = dos_at_ef(calc, 'pdos.gpw', soc=False)
    results['dos_at_ef_soc'] = dos_at_ef(calc, 'pdos.gpw', soc=True)

    # Log key descriptions
    kd = {}
    kd['pdos_nosoc'] = ('Projected density of states '
                        'without spin-orbit coupling '
                        '(PDOS no soc)')
    kd['pdos_soc'] = ('Projected density of states '
                      'with spin-orbit coupling '
                      '(PDOS w. soc)')
    kd['dos_at_ef_nosoc'] = ('KVP: Density of states at the Fermi energy '
                             'without spin-orbit coupling '
                             '(DOS at ef no soc) [states/eV]')
    kd['dos_at_ef_soc'] = ('KVP: Density of states at the Fermi energy '
                           'with spin-orbit coupling '
                           '(DOS at ef w. soc) [states/eV]')
    results.update({'__key_descriptions__': kd})

    return results


# ---------- Recipe methodology ---------- #


# ----- PDOS ----- #


def pdos(calc, gpw, soc=True):
    """Main functionality to do a single pdos calculation"""
    # Do calculation
    e_e, pdos_syl, symbols, ef = calculate_pdos(calc, gpw, soc=soc)

    return {'pdos_syl': pdos_syl, 'symbols': symbols,
            'energies': e_e, 'efermi': ef}


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
    from gpaw.utilities.dos import raw_orbital_LDOS
    from gpaw.utilities.progressbar import ProgressBar
    from ase.parallel import parprint
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
        ldos = raw_spinorbit_orbital_LDOS_hack
    else:
        ldos = raw_orbital_LDOS

    ns = calc.get_number_of_spins()
    theta, phi = get_spin_direction()
    # We want to extract the pdos +-10 eV from efermi
    e_e = np.linspace(-10 + efermi, 10 + efermi, 2000)
    # We distinguish in (spin(s), chemical symbol(y), angular momentum (l)),
    # that is if there are multiple atoms in the unit cell of the same chemical
    # species, their pdos are added together.
    # Set up progressbar
    s_i = [s for s in range(ns) for a in l_a for l in l_a[a]]
    a_i = [a for s in range(ns) for a in l_a for l in l_a[a]]
    l_i = [l for s in range(ns) for a in l_a for l in l_a[a]]
    pdos_syl = defaultdict(float)
    parprint('\nComputing pdos %s' % ('with spin-orbit coupling' * soc))
    pb = ProgressBar()
    for _, (spin, a, l) in pb.enumerate([(s, a, l) for (s, a, l)
                                         in zip(s_i, a_i, l_i)]):
        symbol = chem_symbols[a]

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


def dos_at_ef(calc, gpw, soc=True):
    """Get dos at the Fermi energy"""
    if soc:
        dos = SOCDOS(gpw, width=0.0, window=(-0.1, 0.1), npts=3)
    else:
        dos = DOS(calc, width=0.0, window=(-0.1, 0.1), npts=3)
    return dos.get_dos()[1]


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
    results = 'results-asr.pdos.json'
    pdos = 'pdos_soc' if soc else 'pdos_nosoc'
    if results in row.data and pdos in row.data[results]:
        data = row.data[results][pdos]
    else:
        return

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    # Extract raw data
    symbols = data['symbols']
    pdos_syl = get_ordered_syl_dict(data['pdos_syl'], symbols)
    e_e = data['energies']
    ef = data['efermi']

    # Find energy range to plot in
    gsresults = 'results-asr.gs.json'
    gaps = 'gaps_soc' if soc else 'gaps_nosoc'
    emin = ef - 3
    emax = ef + 3
    if gsresults in row.data and gaps in row.data[gsresults]:
        vbm = row.data[gsresults][gaps]['vbm']
        if vbm is not None:
            emin = vbm - 3
        cbm = row.data[gsresults][gaps]['cbm']
        if cbm is not None:
            emax = cbm + 3

    # Subtract the vacuum energy
    evac = None
    if 'evac' in row.data[gsresults]:
        evac = row.data[gsresults]['evac']
    if evac is not None:
        e_e -= evac
        ef -= evac
        emin -= evac
        emax -= evac

    # Set up energy range to plot in
    i1, i2 = abs(e_e - emin).argmin(), abs(e_e - emax).argmin()

    # Get color code
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


# ---------- ASR main ---------- #

if __name__ == '__main__':
    main.cli()
