from asr.utils import command, option, read_json
import os

@command('asr.hseinterpol')
@option('--kptpath', default=None, type=str)
@option('--npoints', default=400)
def main(kptpath, npoints):
    from asr.hse import bs_interpolate
    results = bs_interpolate(kptpath, npoints)
    return results

def collect_data(atoms):
    """
    Collect results obtained with 2 methods:
    1) ase.dft.kpoints.monkhorst_pack_interpolate
    2) interpolation scheme interpolate_bandlines2(calc, path, e_skn=None)

    XXX
    WARNING: the latter should be ok for 2D, but don't work well for 3D
    we need a better interpolation scheme to work with 3D structures
    """

    kvp = {}
    key_descriptions = {}
    data = {}

    evac = 0.0 # XXX where do I find evac?
    #evac = kvp.get('evac')

    if not os.path.isfile('results_hseinterpol.json'):
        return kvp, key_descriptions, data

    results = read_json('results_hseinterpol.json')

    """
    1) Results obtained with ase.dft.kpoints.monkhorst_pack_interpolate
    """
    dct = results['hse_bandstructure']
     # without soc first
    data['hse_interpol'] = {
        'path': dct['path'],
        'eps_skn': dct['eps_skn'] - evac}
    # then with soc if available
    if 'e_mk' in dct:
        e_mk = dct['e_mk']  # band structure
        s_mk = dct['s_mk']
        data['hse_interpol'].update(eps_mk=e_mk - evac, s_mk=s_mk)

    """
    2) Results from interpolate_bandlines2(calc, path, e_skn=None)
    XXX Warning: not suitable for 3D structures!
    """
    dct = results['hse_bandstructure3']
    if 'epsreal_skn' not in dct:
        warnings.warn('epsreal_skn missing, try and run hseinterpol again')
        return kvp, key_descriptions, data
 
    # without soc first
    data['hse_interpol3'] = {
        'path': dct['path'],
        'eps_skn': dct['eps_skn'] - evac,
        'epsreal_skn': dct['epsreal_skn'] - evac,
        'xkreal': dct['xreal']}
    # then with soc if available
    if 'e_mk' in dct:
        e_mk = dct['e_mk']  # band structure
        data['hse_interpol3'].update(eps_mk=e_mk - evac)

    return kvp, key_descriptions, data


#def bs_hse(row, path):
#    from asr.gw import bs_xc
#    bs_xc(row, path, xc='hse', label='HSE')


"""
# HSE nosoc and HSE soc
def bs_hse(row,
           filename='hse-bs.png',
           figsize=(6.4, 4.8),
           fontsize=10,
           show_legend=True,
           s=0.5):

    if 'hse_interpol' not in row.data:
        return
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    import numpy as np
    from ase.dft.band_structure import BandStructure, BandStructurePlot
    d = row.data.hse_interpol
    e_skn = d['eps_skn']
    nspins = e_skn.shape[0]
    e_kn = np.hstack([e_skn[x] for x in range(nspins)])[np.newaxis]
    path = d['path']
    ef = row.efermi_hse_nosoc
    emin = row.get('vbm_hse', ef) - 3 - ef
    emax = row.get('cbm_hse', ef) + 3 - ef
    mpl.rcParams['font.size'] = fontsize
    bs = BandStructure(path, e_kn, ef)
    # hse without soc
    nosoc_style = dict(
        colors=['0.8'] * e_skn.shape[0],
        label='HSE no SOC',
        ls='-',
        lw=1.0,
        zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    bsp = BandStructurePlot(bs)
    bsp.plot(
        ax=ax,
        show=False,
        emin=emin,
        emax=emax,
        ylabel=r'$E$ [eV]',
        **nosoc_style)
    # hse with soc
    e_mk = d['eps_mk']
    s_mk = d['s_mk']
    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    sdir = row.get('spin_orientation', 'z')
    from asr.bandstructure import plot_with_colors
    ax, cbar = plot_with_colors(
        bsp,
        ax=ax,
        energies=e_mk,
        colors=s_mk,
        filename=filename,
        show=False,
        emin=emin,
        emax=emax,
        sortcolors=True,
        loc='upper right',
        clabel=r'$\langle S_{} \rangle $'.format(sdir),
        s=s)

    ax.set_xlabel('$k$-points')
    plt.legend(loc='upper right')
    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    text = ax.annotate(
        r'$E_\mathrm{F}$',
        xy=(x0, ef),
        ha='left',
        va='bottom',
        fontsize=fontsize * 1.3)
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])
    
    if not show_legend:
        ax.legend_.remove()
    plt.savefig(filename, bbox_inches='tight')
"""

# HSE soc and PBE soc
def bs_hse(row,
           filename='hse-bs.png',
           figsize=(6.4, 4.8),
           fontsize=10,
           show_legend=True,
           s=0.5):

    if 'hse_interpol' not in row.data:
        return

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    import numpy as np
    from ase.dft.band_structure import BandStructure, BandStructurePlot
    from ase.dft.kpoints import labels_from_kpts

    d = row.data.hse_interpol
    path = d['path']
    ef = row.efermi_hse_soc
    emin = row.get('vbm_hse', ef) - 3 - ef
    emax = row.get('cbm_hse', ef) + 3 - ef
    mpl.rcParams['font.size'] = fontsize

    e_mk = d['eps_mk']
    s_mk = d['s_mk']
    x, X, labels = labels_from_kpts(path.kpts, row.cell)
       
    # hse with soc
    hse_style = dict(
        color='k',
        #label='HSE',
        ls='-',
        lw=1.0,
        zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    for e_m in e_mk:
        ax.plot(x, e_m, **hse_style)
    ax.set_ylim([emin, emax])
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylabel(r'$E-E_{\mathrm{vac}}$ [eV]')
    ax.set_xlabel('$k$-points')
    ax.set_xticks(X)
    ax.set_xticklabels([lab.replace('G', r'$\Gamma$') for lab in labels])

    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    ax.axhline(ef, c='k', ls=':')
    text = ax.annotate(
        r'$E_\mathrm{F}$',
        xy=(x0, ef),
        ha='left',
        va='bottom',
        fontsize=fontsize * 1.3)
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    # add PBE band structure with soc
    from asr.bandstructure import add_bs_pbe
    if 'bs_pbe' in row.data and 'path' in row.data.bs_pbe:
        ax = add_bs_pbe(row, ax)
    
    for Xi in X:
        ax.axvline(Xi, ls='-', c='0.5', zorder=-20)

    line_hse = ax.plot([], [], **hse_style, label='HSE')
    plt.legend(loc='upper right')
   
    if not show_legend:
        ax.legend_.remove()
    plt.savefig(filename, bbox_inches='tight')


def webpanel(row, key_descriptions):
    from asr.utils.custom import fig, table
    
    hse = table(row, 'Property',
                ['gap_hse', 'dir_gap_hse', 'vbm_hse', 'cbm_hse'],
                 kd=key_descriptions)

    panel = ('Electronic band structure (HSE)',
             [[fig('hse-bs.png')],
              [hse]])

    things = [(bs_hse, ['hse-bs.png'])]

    return panel, things


group = 'property'
resources = '1:10m'
creates = ['results_hseinterpol.json']
dependencies = ['asr.hse']
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart

if __name__ == '__main__':
    main()
