"""Orbital projected band structure."""
import numpy as np
from ase import Atoms

import asr
from asr.core import (
    command, option, ASRResult, prepare_result, atomsopt, calcopt,
)
import typing

from asr.database.browser import (make_panel_description,
                                  fig, describe_entry, WebPanel)
from asr.bandstructure import calculate as bscalculate, main as bsmain


panel_description = make_panel_description(
    """The single-particle band structure and density of states projected onto
atomic orbitals (s,p,d). Spin–orbit interactions are not included in these
plots.""",
    articles=[
        'C2DB',
    ],
)


scf_projected_bs_filename = 'scf-projected-bs.png'


def webpanel(result, context):
    # XXX Why is it named bandstructure:calculate, why not projected?
    desc1 = context.parameter_description('asr.bandstructure:calculate')
    desc2 = context.parameter_description_picky('asr.gs')

    explanation = ('Orbital projected band structure without '
                   'spin–orbit coupling\n\n' + desc1 + '\n' + desc2)

    # XXX Actually we use the SOC Fermilevel as reference in 3D systems
    # don't we?

    panel = WebPanel(
        title=describe_entry(
            f'Projected band structure and DOS ({context.xcname})',
            panel_description),
        columns=[[describe_entry(fig(scf_projected_bs_filename, link='empty'),
                                 description=explanation)],
                 [fig('bz-with-gaps.png')]],
        plot_descriptions=[{'function': projected_bs_scf,
                            'filenames': [scf_projected_bs_filename]}],
        sort=13.5)

    return [panel]


@prepare_result
class Result(ASRResult):

    symbols: typing.List[str]
    yl_i: typing.List[typing.Tuple[str, str]]
    weight_skni: typing.List[typing.List[typing.List[float]]]

    key_descriptions: typing.Dict[str, str] = dict(
        symbols="Chemical symbols.",
        yl_i="Symbol and orbital angular momentum string ('y,l') of each orbital i.",
        weight_skni="Weight of each projector (indexed by (s, k, n)) on orbitals i.",
    )

    formats = {'webpanel2': webpanel}


# ---------- Main functionality ---------- #

sel = asr.Selector()
sel.version = sel.EQ(-1)
sel.name = sel.EQ('asr.projected_bandstructure:main')


@asr.migration(selector=sel)
def add_bsrestart(record):
    """Add bsrestart parameters."""
    emptybands = (
        record.parameters.dependency_parameters[
            'asr.bandstructure:calculate']['emptybands']
    )
    record.parameters.bsrestart = {
        'nbands': -emptybands,
        'txt': 'bs.txt',
        'fixdensity': True,
        'convergence': {
            'bands': -emptybands // 2},
        'symmetry': 'off'
    }
    del record.parameters.dependency_parameters[
        'asr.bandstructure:calculate']['emptybands']
    return record


@command(
    module='asr.projected_bandstructure',
    migrations=[add_bsrestart],
)
@atomsopt
@calcopt
@asr.calcopt(
    aliases=['-b', '--bsrestart'],
    help='Bandstructure Calculator params.',
    matcher=asr.matchers.EQUAL,
)
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints',
        type=int,
        help='Number of points along k-point path.')
def main(
        atoms: Atoms,
        calculator: dict = bscalculate.defaults.calculator,
        bsrestart: dict = bscalculate.defaults.bsrestart,
        kptpath: typing.Union[str, None] = bscalculate.defaults.kptpath,
        npoints: int = bscalculate.defaults.npoints,
) -> Result:
    # Get bandstructure calculation

    args = dict(
        atoms=atoms,
        calculator=calculator,
        bsrestart=bsrestart,
        kptpath=kptpath,
        npoints=npoints,
    )
    res = bscalculate(**args)

    # We need to depend on bsmain, otherwise we cannot make the web panels
    # since we do not have the requisite metadata.
    bsmain(**args)

    calc = res.calculation.load()
    # calc = GPAW('bs.gpw', txt=None)

    results = {}

    # Extract projections
    weight_skni, yl_i = get_orbital_ldos(calc)
    results['weight_skni'] = weight_skni
    results['yl_i'] = yl_i
    results['symbols'] = list(calc.atoms.symbols)
    return Result.fromdata(**results)


# ---------- Recipe methodology ---------- #


def get_orbital_ldos(calc):
    """Get the projection weights on different orbitals.

    Returns
    -------
    weight_skni : nd.array
        weight of each projector (indexed by (s, k, n)) on orbitals i
    yl_i : list
        symbol and orbital angular momentum string ('y,l') of each orbital i
    """
    from ase.utils import DevNull
    from ase.parallel import parprint
    import gpaw.mpi as mpi
    from gpaw.utilities.dos import raw_orbital_LDOS
    from gpaw.utilities.progressbar import ProgressBar
    from asr.pdos import get_l_a

    ns = calc.get_number_of_spins()
    zs = calc.atoms.get_atomic_numbers()
    atoms = calc.atoms
    l_a = get_l_a(zs)

    # We distinguish in (chemical symbol(y), angular momentum (l)),
    # that is if there are multiple atoms in the unit cell of the same chemical
    # species, their weights are added together.
    # x index for each unique atom
    a_x = [a for a in l_a for l in l_a[a]]
    l_x = [l for a in l_a for l in l_a[a]]
    # Get i index for each unique symbol
    yl_i = []
    i_x = []
    for a, l in zip(a_x, l_x):
        symbol = atoms.symbols[a]
        yl = ','.join([str(symbol), str(l)])
        if yl in yl_i:
            i = yl_i.index(yl)
        else:
            i = len(yl_i)
            yl_i.append(yl)
        i_x.append(i)

    # Allocate output array
    nk, nb = len(calc.get_ibz_k_points()), calc.get_number_of_bands()
    weight_skni = np.zeros((ns, nk, nb, len(yl_i)))

    # Set up progressbar
    ali_x = [(a, l, i) for (a, l, i) in zip(a_x, l_x, i_x)]
    parprint('Computing orbital ldos')
    if mpi.world.rank == 0:
        pb = ProgressBar()
    else:
        devnull = DevNull()
        pb = ProgressBar(devnull)

    for _, (a, l, i) in pb.enumerate(ali_x):
        # Extract weights
        for s in range(ns):
            __, weights = raw_orbital_LDOS(calc, a, s, l)
            weight_kn = weights.reshape((nk, nb))
            # Renormalize (don't include reciprocal space volume in weight)
            weight_kn /= calc.wfs.kd.weight_k[:, np.newaxis]
            weight_skni[s, :, :, i] += weight_kn

    return weight_skni, yl_i


# ---------- Plotting ---------- #


def get_yl_ordering(yl_i, symbols):
    """Get standardized yl ordering of keys.

    Parameters
    ----------
    yl_i : list
        see get_orbital_ldos
    symbols : list
        Sort symbols after index in this list

    Returns
    -------
    c_i : list
        ordered index for each i
    """
    # Setup sili (symbol index, angular momentum index) key
    def sili(yl):
        y, L = yl.split(',')
        # Symbols list can have multiple entries of the same symbol
        # ex. ['O', 'Fe', 'O']. In this case 'O' will have index 0 and
        # 'Fe' will have index 1.
        si = symbols.index(y)
        li = ['s', 'p', 'd', 'f'].index(L)
        return f'{si}{li}'

    i_c = [iyl[0] for iyl in sorted(enumerate(yl_i), key=lambda t: sili(t[1]))]
    return [i_c.index(i) for i in range(len(yl_i))]


def get_bs_sampling(bsp, npoints=40):
    """Sample band structure as evenly as possible.

    Allways include special points.

    Parameters
    ----------
    bsp : obj
        ase.spectrum.band_structure.BandStructurePlot object
    npoints : int
        number of k-points to sample along band structure

    Returns
    -------
    chosenx_x : 1d np.array
        chosen band structure coordinates
    k_x : 1d np.array
        chosen k-point indices
    """
    # Get band structure coordinates and unique labels
    xcoords, label_xcoords, orig_labels = bsp.bs.get_labels()
    label_xcoords = np.unique(label_xcoords)

    # Reserve one point for each special point
    nonspoints = npoints - len(label_xcoords)
    assert nonspoints >= 0
    assert npoints <= len(xcoords)

    # Slice xcoords into seperate subpaths
    xcoords_lx = []
    subpl_l = []
    lastx = 0.
    for labelx in label_xcoords:
        xcoords_x = xcoords[np.logical_and(xcoords >= lastx,
                                           xcoords <= labelx)]
        xcoords_lx.append(xcoords_x)
        subpl_l.append(xcoords_x[-1] - xcoords_x[0])  # Length of subpath
        lastx = labelx

    # Distribute trivial k-points based on length of slices
    pathlength = sum(subpl_l)
    unitlength = pathlength / (nonspoints + 1)
    # Floor npoints and length remainder for each subpath
    subpnp_l, subprl_l = np.divmod(subpl_l, unitlength)
    subpnp_l = subpnp_l.astype(int)
    # Distribute remainders
    points_left = nonspoints - np.sum(subpnp_l)
    subpnp_l[np.argsort(subprl_l)[-points_left:]] += 1

    # Choose points on each sub path
    chosenx_x = []
    for subpnp, xcoords_x in zip(subpnp_l, xcoords_lx):
        # Evenly spaced indices
        x_p = np.unique(np.round(np.linspace(0, len(xcoords_x) - 1,
                                             subpnp + 2)).astype(int))
        chosenx_x += list(xcoords_x[x_p][:-1])  # each subpath includes start
    chosenx_x.append(xcoords[-1])  # Add end of path

    # Get k-indeces
    chosenx_x = np.array(chosenx_x)
    x_y, k_y = np.where(chosenx_x[:, np.newaxis] == xcoords[np.newaxis, :])
    x_x, y_x = np.unique(x_y, return_index=True)
    k_x = k_y[y_x]

    return chosenx_x, k_x


def get_pie_slice(theta0, theta, s=36., res=64):
    """Get a single pie slice marker.

    Parameters
    ----------
    theta0 : float
        angle in which to start slice
    theta : float
        angle that pie slice should cover
    s : float
        marker size
    res : int
        resolution of pie (in points around the circumference)

    Returns
    -------
    pie : matplotlib.pyplot.scatter option dictionary
    """
    assert -np.pi / res <= theta0 and theta0 <= 2. * np.pi + np.pi / res
    assert -np.pi / res <= theta and theta <= 2. * np.pi + np.pi / res

    angles = np.linspace(theta0, theta0 + theta,
                         int(np.ceil(res * theta / (2 * np.pi))))
    x = [0] + np.cos(angles).tolist()
    y = [0] + np.sin(angles).tolist()
    xy = np.column_stack([x, y])
    size = s * np.abs(xy).max() ** 2

    return {'marker': xy, 's': size, 'linewidths': 0.0}


def get_pie_markers(weight_xi, scale_marker=True, s=36., res=64):
    """Get pie markers corresponding to a 2D array of weights.

    Parameters
    ----------
    weight_xi : 2d np.array
    scale_marker : bool
        using sum of weights as scale for markersize
    s, res : see get_pie_slice

    Returns
    -------
    pie_xi : list of lists of mpl option dictionaries
    """
    assert np.all(weight_xi >= 0.)

    pie_xi = []
    for weight_i in weight_xi:
        pie_i = []
        # Normalize by total weight
        totweight = np.sum(weight_i)
        r0 = 0.
        for weight in weight_i:
            # Weight fraction
            r1 = weight / totweight

            # Get slice
            pie = get_pie_slice(2 * np.pi * r0,
                                2 * np.pi * r1, s=s, res=res)
            if scale_marker:
                pie['s'] *= totweight

            pie_i.append(pie)
            r0 += r1
        pie_xi.append(pie_i)

    return pie_xi


def projected_bs_scf(context, filename,
                     npoints=40, markersize=36., res=64,
                     figsize=(5.5, 5), fontsize=10):
    """Produce the projected band structure.

    Plot the projection weight fractions as pie charts on top of the band structure.

    Parameters
    ----------
    npoints : int,
        number of pie charts per band
    markersize : float
        size of pie chart markers
    res : int
        resolution of the pie chart markers (in points around the circumference)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.lines import Line2D
    from ase.spectrum.band_structure import BandStructure, BandStructurePlot

    # Extract projections data
    data = context.result
    weight_skni = data['weight_skni']
    yl_i = data['yl_i']

    # Get color indeces
    c_i = get_yl_ordering(yl_i, data['symbols'])

    # Extract band structure data
    d = context.get_record('asr.bandstructure').result
    path = d['bs_nosoc']['path']
    ef = d['bs_nosoc']['efermi']

    # If a vacuum energy is available, use it as a reference
    eref = context.energy_reference()
    ref = eref.value

    # Determine plotting window based on band gap
    gs_results = context.gs_results()
    gaps = gs_results['gaps_nosoc']
    if gaps.get('vbm'):
        emin = gaps.get('vbm') - 3
    else:
        emin = ef - 3
    if gaps.get('cbm'):
        emax = gaps.get('cbm') + 3
    else:
        emax = ef + 3

    # Take bands with energies in range
    e_skn = d['bs_nosoc']['energies']
    inrange_skn = (e_skn > emin) & (e_skn < emax)
    inrange_n = np.any(np.any(inrange_skn, axis=1), axis=0)
    e_skn = e_skn[:, :, inrange_n]
    weight_skni = weight_skni[:, :, inrange_n, :]

    # Use band structure objects to plot outline
    bs = BandStructure(path, e_skn - ref, ef - ref)
    # Use colors if spin-polarized
    if e_skn.shape[0] == 2:
        spincolors = ['0.8', '0.4']
    else:
        spincolors = ['0.8'] * e_skn.shape[0]
    style = dict(
        colors=spincolors,
        ls='-',
        lw=1.0,
        zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    bsp = BandStructurePlot(bs)
    bsp.plot(ax=ax, show=False, emin=emin - ref, emax=emax - ref,
             ylabel=eref.mpl_plotlabel(), **style)

    xcoords, k_x = get_bs_sampling(bsp, npoints=npoints)

    # Generate energy and weight arrays based on band structure sampling
    ns, nk, nb = e_skn.shape
    s_u = np.array([s for s in range(ns) for n in range(nb)])
    n_u = np.array([n for s in range(ns) for n in range(nb)])
    e_ux = e_skn[s_u[:, np.newaxis],
                 k_x[np.newaxis, :],
                 n_u[:, np.newaxis]] - ref
    weight_uxi = weight_skni[s_u[:, np.newaxis],
                             k_x[np.newaxis, :],
                             n_u[:, np.newaxis], :]
    # Plot projections
    for e_x, weight_xi in zip(e_ux, weight_uxi):

        # Weights as pie chart
        pie_xi = get_pie_markers(weight_xi, s=markersize,
                                 scale_marker=False, res=res)
        for x, e, weight_i, pie_i in zip(xcoords, e_x, weight_xi, pie_xi):
            # totweight = np.sum(weight_i)
            for i, pie in enumerate(pie_i):
                ax.scatter(x, e, facecolor='C{}'.format(c_i[i]),
                           zorder=3, **pie)

    # Set legend
    # Get "pac-man" style pie slice marker
    pie = get_pie_slice(1. * np.pi / 4.,
                        3. * np.pi / 2., s=markersize, res=res)
    # Generate markers for legend
    legend_markers = []
    for i, yl in enumerate(yl_i):
        legend_markers.append(Line2D([0], [0],
                                     mfc='C{}'.format(c_i[i]), mew=0.0,
                                     marker=pie['marker'], ms=3. * np.pi,
                                     linewidth=0.0))
    # Generate legend
    plt.legend(legend_markers, [yl.replace(',', ' (') + ')' for yl in yl_i],
               bbox_to_anchor=(0., 1.02, 1., 0.), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)

    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    text = ax.annotate(
        r'$E_\mathrm{F}$',
        xy=(x0, ef - ref),
        fontsize=mpl.rcParams['font.size'] * 1.25,
        ha='left',
        va='bottom')

    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    # ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    main.cli()
