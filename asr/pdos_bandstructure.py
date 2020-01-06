import numpy as np

from asr.core import command


# ---------- Webpanel ---------- #


def webpanel(row, key_descriptions):
    from asr.database.browser import fig

    panel = {'title': 'Band structure with pdos (PBE)',
             'columns': [[fig('pbe-pdos-bs.png', link='empty')], []],
             'plot_descriptions': [{'function': pdos_bs_pbe,
                                    'filenames': ['pbe-pdos-bs.png']}]}

    return [panel]


# ---------- Main functionality ---------- #


@command(module='asr.pdos_bandstructure',
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


def pdos_bs_pbe(row,
                filename='pbe-pdos-bs.png',
                figsize=(6.4, 4.8),
                fontsize=10):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.dft.band_structure import BandStructure, BandStructurePlot
    mpl.rcParams['font.size'] = fontsize

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

    # Use band structure objects to plot
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

    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    main.cli()
