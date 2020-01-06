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
    return {}


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
