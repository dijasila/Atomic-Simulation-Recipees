def bs_xc(row, path, xc, **kwargs):
    """xc: 'gw' or 'hse'
    """
    from c2db.bsfitfig import bsfitfig
    from asr.bandstructure import add_bs_pbe
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    lw = kwargs.get('lw', 1)
    if row.data.get('bs_pbe', {}).get('path') is None:
        return
    if 'bs_' + xc not in row.data:
        return
    ax = bsfitfig(row, xc=xc, lw=lw)
    if ax is None:
        return
    label = kwargs.get('label', '?')
    # trying to make the legend label look nice
    for line1 in ax.lines:
        if line1.get_marker() == 'o':
            break
    line0 = ax.lines[0]
    line1, = ax.plot([], [],
                     '-o',
                     c=line0.get_color(),
                     markerfacecolor=line1.get_markerfacecolor(),
                     markeredgecolor=line1.get_markeredgecolor(),
                     markersize=line1.get_markersize(),
                     lw=line0.get_lw())
    line1.set_label(label)
    if 'bs_pbe' in row.data and 'path' in row.data.bs_pbe:
        ax = add_bs_pbe(row, ax, **kwargs)
    ef = row.get('efermi_{}'.format(xc))
    ax.axhline(ef, c='k', ls=':')
    emin = row.get('vbm_' + xc, ef) - 3
    emax = row.get('cbm_' + xc, ef) + 3
    ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
    ax.set_ylim(emin, emax)
    ax.set_xlabel('$k$-points')
    leg = ax.legend(loc='upper right')
    leg.get_frame().set_alpha(1)
    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    text = ax.annotate(
        r'$E_\mathrm{F}$', xy=(x0, ef), ha='left', va='bottom', fontsize=13)
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])
    plt.savefig(path)
    plt.close()


def bs_gw(row, path):
    bs_xc(row, path, xc='gw', label='G$_0$W$_0$')


# def webpanel(row, key_descriptions):
#     from asr.custom import fig, table
#     gw = table(
#         'Property', ['gap_gw', 'dir_gap_gw', 'vbm_gw', 'cbm_gw'],
#         key_descriptions=key_descriptions_noxc)

#     panel = ('Electronic band structure (GW)', [[fig('gw-bs.png')], [gw]])

#     things = [(bs_gw, ['gw-bs.png']), (bs_hse, ['hse-bs.png'])]

#     return panel, things


group = 'Property'
