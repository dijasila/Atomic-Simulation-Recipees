from asr.core import command, option, read_json

from collections import defaultdict

import numpy as np

from ase import Atoms
from ase.units import Ha
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset as k2so
from ase.dft.dos import DOS
from ase.dft.dos import linear_tetrahedron_integration as lti

from ase.units import Hartree

from asr.utils import magnetic_atoms


# ---------- GPAW hacks ---------- #


# Hack the density of states
class SOCDOS(DOS):
    def __init__(self, gpw, npts=401, **kwargs):
        """Hack to make DOS class work with spin orbit coupling.

        Parameters
        ----------
        gpw : str
            The SOCDOS takes a filename of the GPAW calculator object and loads
            it, instead of the normal ASE compliant calculator object.
        npts : int
            see ase.dft.dos.DOS

        """
        # Initiate DOS with serial communicator instead
        from gpaw import GPAW
        import gpaw.mpi as mpi
        from asr.utils.gpw2eigs import calc2eigs
        from asr.magnetic_anisotropy import get_spin_axis

        # Only the rank=0 should have an actual DOS object.
        # The others receive the output as a broadcast.
        self.world = mpi.world
        if self.world.rank == 0:
            # Initiate calculator object and get the spin-orbit eigenvalues
            calc0 = GPAW(gpw, communicator=mpi.serial_comm, txt=None)
            theta, phi = get_spin_axis()
            e_skm, ef = calc2eigs(calc0, theta=theta, phi=phi, ranks=[0])

            DOS.__init__(self, calc0, npts=npts, comm=calc0.world, **kwargs)

            # Hack the number of spins
            self.nspins = 1

            # Hack the eigenvalues
            if e_skm.ndim == 2:
                e_skm = e_skm[np.newaxis]
            e_skn = e_skm - ef
            bzkpts = calc0.get_bz_k_points()
            size, offset = k2so(bzkpts)
            bz2ibz = calc0.get_bz_to_ibz_map()
            shape = (self.nspins, ) + tuple(size) + (-1, )
            self.e_skn = e_skn[:, bz2ibz].reshape(shape)
        else:
            self.npts = npts

    def get_dos(self):
        """Interface to DOS.get_dos."""
        # Rank=0 calculates the dos
        if self.world.rank == 0:
            dos = np.ascontiguousarray(DOS.get_dos(self, spin=0))
        else:
            dos = np.empty(self.npts)

        # Broadcast result
        self.world.broadcast(dos, 0)

        return dos


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
        from gpaw.spinorbit import get_spinorbit_eigenvalues
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
    """Hack raw_spinorbit_orbital_LDOS."""
    from gpaw.utilities.dos import get_angular_projectors
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
            weights_xi[x:x + nb, :] += w * np.absolute(P_ami[a][:,
                                                                1::2])**2 / 2
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

params = "{'mode':{'ecut':200,...},'kpts':{'density':2.0},...}"
ctests = []
ctests.append({'description': 'Test the refined ground state of Si',
               'name': 'test_asr.pdos_Si_gpw',
               'tags': ['gitlab-ci'],
               'cli': ['asr run "setup.materials -s Si2"',
                       'ase convert materials.json structure.json',
                       'asr run "setup.params '
                       f'asr.gs@calculate:calculator {params} '
                       'asr.pdos@calculate:kptdensity 3.0 '
                       'asr.pdos@calculate:emptybands 5"',
                       'asr run gs',
                       'asr run pdos@calculate',
                       'asr run database.fromtree',
                       'asr run "database.browser --only-figures"']})

tests = []
tests.append({'description': 'Test the pdos of Si (cores=1)',
              'name': 'test_asr.pdos_Si_serial',
              'cli': ['asr run "setup.materials -s Si2"',
                      'ase convert materials.json structure.json',
                      'asr run "setup.params '
                      f'asr.gs@calculate:calculator {params} '
                      'asr.pdos@calculate:kptdensity 3.0 '
                      'asr.pdos@calculate:emptybands 5"',
                      'asr run gs',
                      'asr run pdos',
                      'asr run database.fromtree',
                      'asr run "database.browser --only-figures"']})
tests.append({'description': 'Test the pdos of Si (cores=2)',
              'name': 'test_asr.pdos_Si_parallel',
              'cli': ['asr run "setup.materials -s Si2"',
                      'ase convert materials.json structure.json',
                      'asr run "setup.params '
                      f'asr.gs@calculate:calculator {params} '
                      'asr.pdos@calculate:kptdensity 3.0 '
                      'asr.pdos@calculate:emptybands 5"',
                      'asr run gs',
                      'asr run -p 2 pdos',
                      'asr run database.fromtree',
                      'asr run "database.browser --only-figures"']})


# ---------- Webpanel ---------- #


def webpanel(row, key_descriptions):
    from asr.database.browser import fig
    # PDOS without spin-orbit coupling
    panel = {'title': 'Projected band structure and DOS (PBE)',
             'columns': [[],
                         [fig('pbe-pdos_nosoc.png', link='empty')]],
             'plot_descriptions': [{'function': plot_pdos_nosoc,
                                    'filenames': ['pbe-pdos_nosoc.png']}],
             'sort': 13}

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
def calculate(kptdensity: float = 20.0, emptybands: int = 20):
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
    from asr.core import singleprec_dict
    from ase.parallel import parprint

    # Get refined ground state with more k-points
    calc = GPAW('pdos.gpw', txt=None)

    results = {}

    # Calculate the dos at the Fermi energy
    parprint('\nComputing dos at Ef', flush=True)
    results['dos_at_ef_nosoc'] = dos_at_ef(calc, 'pdos.gpw', soc=False)
    parprint('\nComputing dos at Ef with spin-orbit coupling', flush=True)
    results['dos_at_ef_soc'] = dos_at_ef(calc, 'pdos.gpw', soc=True)

    # Calculate pdos
    parprint('\nComputing pdos', flush=True)
    results['pdos_nosoc'] = singleprec_dict(pdos(calc, 'pdos.gpw', soc=False))
    parprint('\nComputing pdos with spin-orbit coupling', flush=True)
    results['pdos_soc'] = singleprec_dict(pdos(calc, 'pdos.gpw', soc=True))

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
    """Do a single pdos calculation.

    Main functionality to do a single pdos calculation.
    """
    # Do calculation
    e_e, pdos_syl, symbols, ef = calculate_pdos(calc, gpw, soc=soc)

    return {'pdos_syl': pdos_syl, 'symbols': symbols,
            'energies': e_e, 'efermi': ef}


def calculate_pdos(calc, gpw, soc=True):
    """Calculate the projected density of states.

    Returns
    -------
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
    from ase.utils import DevNull
    from asr.magnetic_anisotropy import get_spin_axis
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
    theta, phi = get_spin_axis()
    gaps = read_json('results-asr.gs.json').get('gaps_nosoc')
    e1 = gaps.get('vbm') or gaps.get('efermi')
    e2 = gaps.get('cbm') or gaps.get('efermi')
    e_e = np.linspace(e1 - 3, e2 + 3, 500)

    # We distinguish in (spin(s), chemical symbol(y), angular momentum (l)),
    # that is if there are multiple atoms in the unit cell of the same chemical
    # species, their pdos are added together.
    pdos_syl = defaultdict(float)
    # Set up progressbar
    s_i = [s for s in range(ns) for a in l_a for l in l_a[a]]
    a_i = [a for s in range(ns) for a in l_a for l in l_a[a]]
    l_i = [l for s in range(ns) for a in l_a for l in l_a[a]]
    sal_i = [(s, a, l) for (s, a, l) in zip(s_i, a_i, l_i)]
    if mpi.world.rank == 0:
        pb = ProgressBar()
    else:
        devnull = DevNull()
        pb = ProgressBar(devnull)
    for _, (spin, a, l) in pb.enumerate(sal_i):
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
    """Define which atoms and angular momentum to project onto.

    Parameters
    ----------
    zs : [z1, z2, ...]-list or array
        list of atomic numbers (zi: int)

    Returns
    -------
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
    """Get dos at the Fermi energy."""
    if soc:
        dos = SOCDOS(gpw, width=0.0, window=(-0.1, 0.1), npts=3)
    else:
        dos = DOS(calc, width=0.0, window=(-0.1, 0.1), npts=3)
    return dos.get_dos()[1]


# ---------- Plotting ---------- #


def get_ordered_syl_dict(dct_syl, symbols):
    """Order a dictionary with syl keys.

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
        s, y, L = syl.split(',')
        # Symbols list can have multiple entries of the same symbol
        # ex. ['O', 'Fe', 'O']. In this case 'O' will have index 0 and
        # 'Fe' will have index 1.
        si = symbols.index(y)
        li = ['s', 'p', 'd', 'f'].index(L)
        return f'{s}{si}{li}'

    return OrderedDict(sorted(dct_syl.items(), key=lambda t: ssili(t[0])))


def get_yl_colors(dct_syl):
    """Get the color indices corresponding to each symbol and angular momentum.

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


def plot_pdos_nosoc(*args, **kwargs):
    return plot_pdos(*args, soc=False, **kwargs)


def plot_pdos_soc(*args, **kwargs):
    return plot_pdos(*args, soc=True, **kwargs)


def plot_pdos(row, filename, soc=True,
              figsize=(5.5, 5), lw=1):

    def smooth(y, npts=3):
        return np.convolve(y, np.ones(npts) / npts, mode='same')

    # Check if pdos data is stored in row
    results = 'results-asr.pdos.json'
    pdos = 'pdos_soc' if soc else 'pdos_nosoc'
    if results in row.data and pdos in row.data[results]:
        data = row.data[results][pdos]
    else:
        return

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    # Extract raw data
    symbols = data['symbols']
    pdos_syl = get_ordered_syl_dict(data['pdos_syl'], symbols)
    e_e = data['energies'].copy() - row.get('evac', 0)
    ef = data['efermi']

    # Find energy range to plot in
    if soc:
        emin = row.get('vbm', ef) - 3 - row.get('evac', 0)
        emax = row.get('cbm', ef) + 3 - row.get('evac', 0)
    else:
        nosoc_data = row.data['results-asr.gs.json']['gaps_nosoc']
        vbmnosoc = nosoc_data.get('vbm', ef)
        cbmnosoc = nosoc_data.get('cbm', ef)

        if vbmnosoc is None:
            vbmnosoc = ef

        if cbmnosoc is None:
            cbmnosoc = ef

        emin = vbmnosoc - 3 - row.get('evac', 0)
        emax = cbmnosoc + 3 - row.get('evac', 0)

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
    plt.figure(figsize=figsize)
    ax = plt.gca()

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
                label=label, color=color_yl[key[2:]])

    ax.axhline(ef - row.get('evac', 0), color='k', ls=':')

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
    text = plt.text(x0, ef - row.get('evac', 0),
                    r'$E_\mathrm{F}$',
                    ha='left',
                    va='bottom')

    text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    ax.set_xlabel('projected dos [states / eV]')
    if row.get('evac') is not None:
        ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
    else:
        ax.set_ylabel(r'$E$ [eV]')

    # Set up legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main.cli()
