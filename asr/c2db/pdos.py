"""Projected density of states."""
from pathlib import Path
from asr.core import (
    command, option, ASRResult,
    prepare_result, atomsopt, calcopt)
from asr.c2db.gs import calculate as gscalculate
# from asr.c2db.gs import main as gsmain
from collections import defaultdict
import typing

import numpy as np
from ase import Atoms

from asr.utils import magnetic_atoms


def webpanel(result, context):
    from asr.database.browser import fig, describe_entry, WebPanel

    desc = '\n'.join([
        context.parameter_description('asr.c2db.pdos:calculate'),
        context.parameter_description_picky('asr.c2db.gs:calculate')
    ])

    explanation = ('Orbital projected density of states without spin–orbit '
                   'coupling\n\n' + desc)

    # Projected band structure and DOS panel
    panel = WebPanel(
        title=f'Projected band structure and DOS ({context.xcname})',
        columns=[[],
                 [describe_entry(fig(pdos_figfile, link='empty'),
                                 description=explanation)]],
        plot_descriptions=[{'function': plot_pdos_nosoc,
                            'filenames': [pdos_figfile]}],
        sort=13)

    return [panel]


pdos_figfile = 'scf-pdos_nosoc.png'

# ---------- Main functionality ---------- #


# ----- Slow steps ----- #


@command(module='asr.c2db.pdos')
@atomsopt
@calcopt
@option('-k', '--kptdensity', type=float, help='K-point density')
@option('--emptybands', type=int, help='number of empty bands to include')
def calculate(
        atoms: Atoms,
        calculator: dict = gscalculate.defaults.calculator,
        kptdensity: float = 20.0,
        emptybands: int = 20,
) -> ASRResult:
    from asr.utils.refinegs import refinegs

    calc = refinegs(
        atoms=atoms,
        calculator=calculator,
        kptdensity=kptdensity, emptybands=emptybands,
        txt='pdos.txt',
    )

    gpw = Path('pdos.gpw')
    calc.write(gpw)
    return gpw

# ----- Fast steps ----- #


@prepare_result
class PdosResult(ASRResult):

    efermi: float
    symbols: typing.List[str]
    energies: typing.List[float]
    pdos_syl: typing.List[float]

    key_descriptions: typing.Dict[str, str] = dict(
        efermi="Fermi level [eV] of ground state with dense k-mesh.",
        symbols="Chemical symbols.",
        energies="Energy mesh of pdos results.",
        pdos_syl=("Projected density of states [states / eV] for every set of keys "
                  "'s,y,l', that is spin, symbol and orbital l-quantum number.")
    )


@prepare_result
class Result(ASRResult):

    dos_at_ef_nosoc: float
    dos_at_ef_soc: float
    pdos_nosoc: PdosResult
    pdos_soc: PdosResult

    key_descriptions: typing.Dict[str, str] = dict(
        dos_at_ef_nosoc=("Density of states at the Fermi "
                         "level w/o soc [states / (unit cell * eV)]"),
        dos_at_ef_soc=("Density of states at the Fermi "
                       "level [states / (unit cell * eV)])"),
        pdos_nosoc="Projected density of states w/o soc.",
        pdos_soc="Projected density of states"
    )
    formats = {"webpanel2": webpanel}


@command(module='asr.c2db.pdos')
#@option('-k', '--kptdensity', type=float, help='K-point density')
#@option('--emptybands', type=int, help='number of empty bands to include')
def postprocess(
        gpwfile,  # gpwfile would typically come from pdos calculate()
        mag_ani,
        erange,
) -> Result:
    from gpaw import GPAW
    from ase.parallel import parprint
    calc = GPAW(gpwfile)
    atoms = calc.get_atoms()

    dos1 = calc.dos(shift_fermi_level=False)
    theta, phi = mag_ani.spin_angles()
    dos2 = calc.dos(soc=True, theta=theta, phi=phi, shift_fermi_level=False)

    results = {}

    # Calculate the dos at the Fermi energy
    parprint('\nComputing dos at Ef', flush=True)
    results['dos_at_ef_nosoc'] = dos1.raw_dos([dos1.fermi_level],
                                              width=0.0)[0]
    parprint('\nComputing dos at Ef with spin-orbit coupling', flush=True)
    results['dos_at_ef_soc'] = dos2.raw_dos([dos2.fermi_level],
                                            width=0.0)[0]

    # Calculate pdos
    parprint('\nComputing pdos', flush=True)
    results['pdos_nosoc'] = pdos(atoms, dos1, calc, erange)
    parprint('\nComputing pdos with spin-orbit coupling', flush=True)
    results['pdos_soc'] = pdos(atoms, dos2, calc, erange)

    return Result(results)


class PDOS:
    def __init__(self, atoms, calculator=gscalculate.defaults.calculator,
                 kptdensity=20.0, emptybands=20):
        from asr.c2db.gs import GS
        gs = GS(atoms=atoms, calculator=calculator)
        gaps = gs.post.gaps_nosoc
        e1 = gaps.get('vbm') or gaps.get('efermi')
        e2 = gaps.get('cbm') or gaps.get('efermi')
        erange = np.linspace(e1 - 3, e2 + 3, 500)

        self.pdos_gpwfile = calculate(atoms=atoms, calculator=calculator,
                                      kptdensity=kptdensity, emptybands=emptybands)
        self.post = postprocess(gpwfile=self.pdos_gpwfile,
                                mag_ani=gs.mag_ani, erange=erange)


# ---------- Recipe methodology ---------- #


# ----- PDOS ----- #


def pdos(atoms, dos, calc, erange):
    """Do a single pdos calculation.

    Main functionality to do a single pdos calculation.
    """
    from asr.core import singleprec_dict

    # Do calculation
    e_e, pdos_syl, symbols, ef = calculate_pdos(atoms, dos, calc, erange)

    return PdosResult.fromdata(
        efermi=ef,
        symbols=symbols,
        energies=e_e,
        pdos_syl=singleprec_dict(pdos_syl))


def calculate_pdos(atoms, dos, calc, erange):
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
    import gpaw.mpi as mpi
    from gpaw.utilities.progressbar import ProgressBar
    from ase.utils import DevNull

    zs = calc.atoms.get_atomic_numbers()
    atoms = calc.atoms
    efermi = calc.get_fermi_level()
    l_a = get_l_a(zs)

    ns = calc.get_number_of_spins()
    e_e = erange  # np.linspace(e1 - 3, e2 + 3, 500)

    # We distinguish in (spin(s), chemical symbol(y), angular momentum (l)),
    # that is if there are multiple atoms in the unit cell of the same chemical
    # species, their pdos are added together.
    pdos_syl = defaultdict(float)
    s_i = [s for s in range(ns) for a in l_a for l in l_a[a]]
    a_i = [a for s in range(ns) for a in l_a for l in l_a[a]]
    l_i = [l for s in range(ns) for a in l_a for l in l_a[a]]
    sal_i = [(s, a, l) for (s, a, l) in zip(s_i, a_i, l_i)]

    # Set up progressbar
    if mpi.world.rank == 0:
        pb = ProgressBar()
    else:
        devnull = DevNull()
        pb = ProgressBar(devnull)

    for _, (spin, a, l) in pb.enumerate(sal_i):
        symbol = atoms.symbols[a]

        p = dos.raw_pdos(e_e, a, 'spdfg'.index(l), None, spin, 0.0)

        # Store in dictionary
        key = ','.join([str(spin), str(symbol), str(l)])
        pdos_syl[key] += p

    return e_e, pdos_syl, list(atoms.symbols), efermi


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


def plot_pdos_nosoc(context, *args, **kwargs):
    return plot_pdos(context, *args, soc=False, **kwargs)


def plot_pdos_soc(context, *args, **kwargs):
    return plot_pdos(context, *args, soc=True, **kwargs)


def plot_pdos(context, filename, soc=True,
              figsize=(5.5, 5), lw=1):

    def smooth(y, npts=3):
        return np.convolve(y, np.ones(npts) / npts, mode='same')

    pdos_name = 'pdos_soc' if soc else 'pdos_nosoc'
    result = context.result
    pdos = result[pdos_name]

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.patheffects as path_effects

    ref = context.energy_reference()
    eref = ref.value

    # Extract raw data
    pdos_syl = get_ordered_syl_dict(pdos['pdos_syl'], context.atoms.symbols)
    e_e = pdos['energies'] - eref
    ef = pdos['efermi']

    gs_results = context.gs_results()

    # Find energy range to plot in
    if soc:
        emin = gs_results.get('vbm', ef) - 3 - eref
        emax = gs_results.get('cbm', ef) + 3 - eref
    else:
        nosoc_data = gs_results['gaps_nosoc']
        vbmnosoc = nosoc_data.get('vbm', ef)
        cbmnosoc = nosoc_data.get('cbm', ef)

        if vbmnosoc is None:
            vbmnosoc = ef

        if cbmnosoc is None:
            cbmnosoc = ef

        emin = vbmnosoc - 3 - eref
        emax = cbmnosoc + 3 - eref

    # Set up energy range to plot in
    i1, i2 = abs(e_e - emin).argmin(), abs(e_e - emax).argmin()

    # Get color code
    color_yl = get_yl_colors(pdos_syl)

    # Figure out if pdos has been calculated for more than one spin channel
    spinpol = False
    for k in pdos_syl:
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

    ax.axhline(ef - eref, color='k', ls=':')

    # Set up axis limits
    ax.set_ylim(emin, emax)
    if spinpol:  # Use symmetric limits
        xmax = max(pdosint_s.values())
        ax.set_xlim(-xmax * 0.5, xmax * 0.5)
    else:
        ax.set_xlim(0, pdosint_s[0] * 0.5)

    # Annotate E_F
    xlim = ax.get_xlim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.99
    text = plt.text(x0, ef - eref,
                    r'$E_\mathrm{F}$',
                    fontsize=rcParams['font.size'] * 1.25,
                    ha='right',
                    va='bottom')

    text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    ax.set_xlabel('Projected DOS [states / eV]')
    ax.set_ylabel(ref.mpl_plotlabel())

    # Set up legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()
