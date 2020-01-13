import numpy as np

from asr.core import command


# ---------- Webpanel ---------- #


def webpanel(row, key_descriptions):
    from asr.database.browser import fig

    panel = {'title': 'Band structure with projections (PBE)',
             'columns': [[fig('pbe-projected-bs.png', link='empty')], []],
             'plot_descriptions': [{'function': projected_bs_pbe,
                                    'filenames': ['pbe-projected-bs.png']}]}

    return [panel]


# ---------- Main functionality ---------- #


@command(module='asr.projected_bandstructure',
         requires=['results-asr.gs.json', 'bs.gpw',
                   'results-asr.bandstructure.json'],
         dependencies=['asr.gs', 'asr.bandstructure'],
         webpanel=webpanel)
def main():
    from gpaw import GPAW

    # Get bandstructure calculation
    calc = GPAW('bs.gpw', txt=None)

    results = {}

    # Extract projections
    weight_skni, yl_i = get_orbital_ldos(calc)
    results['weight_skni'] = weight_skni
    results['yl_i'] = yl_i
    results['symbols'] = calc.atoms.get_chemical_symbols()
    
    return results


# ---------- Recipe methodology ---------- #


def get_orbital_ldos(calc):
    """Get the projection weights on different orbitals

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
    chem_symbols = calc.atoms.get_chemical_symbols()
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
        symbol = chem_symbols[a]
        yl = ','.join([str(symbol), str(l)])
        if yl in yl_i:
            i = yl_i.index(yl)
        else:
            i = len(yl_i)
            yl_i.append(yl)
        i_x.append(i)

    # Allocate output array
    nk, nb = calc.wfs.kd.nibzkpts, calc.wfs.bd.nbands
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
    """Get standardized yl ordering of keys

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


def get_pie_markers(weight_xi, s=36., scale_marker=True, res=126):
    """Get pie markers corresponding to a 2D array of weights.

    Parameters
    ----------
    weight_xi : 2d np.array
    s : float
        marker size
    scale_marker : bool
        using sum of weights as scale for markersize
    res : int
        resolution of pie (in points around the circumference)

    Returns
    -------
    pie_ki : list of lists of mpl option dictionaries
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
            rp = np.ceil(r1 * res)
            # Calculate points of the pie marker
            x = [0] + np.cos(np.linspace(2 * np.pi * r0,
                                         2 * np.pi * (r0 + r1), rp)).tolist()
            y = [0] + np.sin(np.linspace(2 * np.pi * r0,
                                         2 * np.pi * (r0 + r1), rp)).tolist()
            xy = np.column_stack([x, y])
            size = totweight * s * np.abs(xy).max() ** 2
            pie_i.append({'marker': xy, 's': size, 'linewidths': 0.0})
            r0 += r1
        pie_xi.append(pie_i)

    return pie_xi


def projected_bs_pbe(row,
                     filename='pbe-projected-bs.png',
                     figsize=(6.4, 4.8),
                     fontsize=10):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.dft.band_structure import BandStructure, BandStructurePlot
    mpl.rcParams['font.size'] = fontsize

    # Extract projections data
    data = row.data.get('results-asr.projected_bandstructure.json')
    weight_skni = data['weight_skni']
    yl_i = data['yl_i']

    # Get color indeces
    c_i = get_yl_ordering(yl_i, data['symbols'])

    # Extract band structure data
    d = row.data.get('results-asr.bandstructure.json')
    path = d['bs_nosoc']['path']
    ef = d['bs_nosoc']['efermi']

    # If a vacuum energy is available, use it as a reference
    ref = row.get('evac', d.get('bs_nosoc').get('efermi'))
    if row.get('evac') is not None:
        label = r'$E - E_\mathrm{vac}$ [eV]'
    else:
        label = r'$E - E_\mathrm{F}$ [eV]'

    # Determine plotting window based on band gap
    gaps = row.data.get('results-asr.gs.json', {}).get('gaps_nosoc', {})
    if gaps.get('vbm'):
        emin = gaps.get('vbm') - 3
    else:
        emin = ef - 3
    if gaps.get('cbm'):
        emax = gaps.get('cbm') + 3
    else:
        emax = ef + 3

    # hstack spin index for the BandStructure object
    e_skn = d['bs_nosoc']['energies']
    nspins = e_skn.shape[0]
    e_kn = np.hstack([e_skn[x] for x in range(nspins)])[np.newaxis]

    # Use band structure objects to plot outline
    bs = BandStructure(path, e_kn - ref, ef - ref)
    style = dict(
        colors=['0.8'] * e_skn.shape[0],
        ls='-',
        lw=1.0,
        zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    bsp = BandStructurePlot(bs)
    bsp.plot(ax=ax, show=False, emin=emin - ref, emax=emax - ref,
             ylabel=label, **style)

    # Choose some plotting format                                              XXX
    # Energy and weight arrays
    ns, nk, nb = e_skn.shape
    s_u = [s for s in range(ns) for n in range(nb)]
    n_u = [n for s in range(ns) for n in range(nb)]
    e_uk = e_skn[s_u, :, n_u] - ref
    weight_uki = weight_skni[s_u, :, n_u, :]
    # Plot projections
    markersize = 36.
    for e_k, weight_ki in zip(e_uk, weight_uki):

        # Marker size from total weight, weights as pie chart
        pie_ki = get_pie_markers(weight_ki, s=markersize)
        for x, e, weight_i, pie_i in zip(bsp.xcoords, e_k, weight_ki, pie_ki):
            totweight = np.sum(weight_i)
            for i, pie in enumerate(pie_i):
                ax.scatter(x, e, facecolor='C{}'.format(c_i[i]),
                           zorder=3, **pie)

        # Marker size depending on each weight
        # for x, e, weight_i in zip(bsp.xcoords, e_k, weight_ki):
        #     # Sort orbital after weight
        #     i_si = np.argsort(weight_i)
        #     # Calculate accumulated weight and use it for area of marker.
        #     # Join orbitals with less than 10% weight in a neutral marker
        #     totweight = np.sum(weight_i)
        #     accweight = 0.
        #     neutralweight = 0.
        #     c_pi = []
        #     a_pi = []
        #     for i, weight in zip(i_si, weight_i[i_si]):
        #         if weight < 0.1 * totweight:
        #             neutralweight += weight
        #         else:
        #             if neutralweight is not None:
        #                 # Done with small weights, store neutral marker
        #                 a_pi.append(markersize * neutralweight)
        #                 c_pi.append('xkcd:grey')
        #                 accweight += neutralweight
        #                 neutralweight = None
        #             # Add orbital as is
        #             accweight += weight
        #             a_pi.append(markersize * accweight)
        #             c_pi.append('C{}'.format(c_i[i]))
        #     # Plot points from largest to smallest
        #     for a, c in zip(reversed(a_pi), reversed(c_pi)):
        #         ax.scatter(x, e, color=c, s=a, zorder=3)

        # Marker color depending on largest weight
        # c_k = [c_i[i] for i in np.argmax(weight_ki, axis=1)]
        # for x, e, c in zip(bsp.xcoords, e_k, c_k):
        #     ax.scatter(x, e, color='C{}'.format(c), s=markersize, zorder=3)

    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    main.cli()
