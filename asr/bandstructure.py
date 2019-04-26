from asr.utils import option, update_defaults
import click


@click.command()
@update_defaults('asr.bandstructure')
@option('--kptpath', default=None, type=str)
@option('--npoints', default=400)
@option('--emptybands', default=20)
def main(kptpath, npoints, emptybands):
    """Calculate electronic band structure"""
    import os
    from pathlib import Path
    import json
    from ase.io import jsonio

    from gpaw import GPAW
    import gpaw.mpi as mpi
    from ase.io import read
    from ase.dft.band_structure import get_band_structure

    if os.path.isfile('eigs_spinorbit.npz'):
        return
    assert os.path.isfile('gs.gpw')

    ref = GPAW('gs.gpw', txt=None).get_fermi_level()

    if kptpath is None:
        atoms = read('gs.gpw')
        kptpath = atoms.cell.bandpath(npoints=npoints)

    if not os.path.isfile('bs.gpw'):
        convbands = emptybands // 2
        parms = {
            'basis': 'dzp',
            'nbands': -emptybands,
            'txt': 'bs.txt',
            'fixdensity': True,
            'kpts': kptpath,
            'convergence': {
                'bands': -convbands},
            'symmetry': 'off'}

        calc = GPAW('gs.gpw', **parms)
        calc.get_potential_energy()
        calc.write('bs.gpw')

    calc = GPAW('bs.gpw', txt=None)
    bs = get_band_structure(calc=calc, _bandpath=kptpath, _reference=ref)

    if mpi.world.rank == 0:
        data = bs.todict()
        data = jsonio.encode(data)

        Path('results-bs-nosoc.json').write_text(data)

    # stuff below could be moved to the collect script.
    e_km, _, s_kvm = gpw2eigs(
        'bs.gpw', soc=True, return_spin=True, optimal_spin_direction=True)

    if mpi.world.rank == 0:
        data = bs.todict()
        data['energies'] = e_km.T
        data['spin_mvk'] = s_kvm.transpose(2, 1, 0)

        data = jsonio.encode(data)
        Path('results-bs-soc.json').write_text(json.dumps(data))


def gpw2eigs(gpw, soc=True, bands=None, return_spin=False,
             optimal_spin_direction=False):
    """give the eigenvalues w or w/o spinorbit coupling and the corresponding
    fermi energy
    Parameters:
        gpw: str
            gpw filename
        soc: None, bool
            use spinorbit coupling if None it returns both w and w/o
        optimal_spin_direction: bool
            If True, use get_spin_direction to calculate the spin direction
            for the SOC
        bands: slice, list of ints or None
            None gives parameters.convergence.bands if possible else all bands
        Returns: dict or e_skn, efermi
        containg eigenvalues and fermi levels w and w/o spinorbit coupling
    """
    import numpy as np
    from gpaw import GPAW, mpi
    from gpaw.spinorbit import get_spinorbit_eigenvalues
    from ase.parallel import broadcast
    ranks = [0]
    comm = mpi.world.new_communicator(ranks)
    dct = None
    if mpi.world.rank in ranks:
        theta = 0
        phi = 0
        if optimal_spin_direction:
            theta, phi = get_spin_direction()
        calc = GPAW(gpw, txt=None, communicator=comm)
        if bands is None:
            n2 = calc.todict().get('convergence', {}).get('bands')
            bands = slice(0, n2)
        if isinstance(bands, slice):
            bands = range(calc.get_number_of_bands())[bands]
        eps_nosoc_skn = eigenvalues(calc)[..., bands]
        efermi_nosoc = calc.get_fermi_level()
        eps_mk, s_kvm = get_spinorbit_eigenvalues(calc, bands=bands,
                                                  theta=theta,
                                                  phi=phi,
                                                  return_spin=True)
        eps_km = eps_mk.T
        efermi = fermi_level(calc, eps_km[np.newaxis],
                             nelectrons=2 *
                             calc.get_number_of_electrons())
        dct = {'eps_nosoc_skn': eps_nosoc_skn,
               'eps_km': eps_km,
               'efermi_nosoc': efermi_nosoc,
               'efermi': efermi,
               's_kvm': s_kvm}

    dct = broadcast(dct, root=0, comm=mpi.world)
    if soc is None:
        return dct
    elif soc:
        out = (dct['eps_km'], dct['efermi'], dct['s_kvm'])
        if not return_spin:
            out = out[:2]
        return out
    else:
        return dct['eps_nosoc_skn'], dct['efermi_nosoc']


def fermi_level(calc, eps_skn=None, nelectrons=None):
    """
    Parameters:
        calc: GPAW
            GPAW calculator
        eps_skn: ndarray, shape=(ns, nk, nb), optional
            eigenvalues (taken from calc if None)
        nelectrons: float, optional
            number of electrons (taken from calc if None)
    Returns:
        out: float
            fermi level
    """
    from ase.units import Ha
    from gpaw.occupations import occupation_numbers
    if nelectrons is None:
        nelectrons = calc.get_number_of_electrons()
    if eps_skn is None:
        eps_skn = eigenvalues(calc)
    eps_skn.sort(axis=-1)
    occ = calc.occupations.todict()
    weight_k = calc.get_k_point_weights()
    return occupation_numbers(occ, eps_skn, weight_k, nelectrons)[1] * Ha


def is_symmetry_protected(kpt, op_scc):
    """Calculate electronic band structure"""
    import numpy as np

    mirror_count = 0
    for symm in op_scc:
        # Inversion symmetry forces spin degeneracy and 180 degree rotation
        # forces the spins to lie in plane
        if (np.allclose(symm, -1 * np.eye(3))
                or np.allclose(symm, np.array([-1, -1, 1] * np.eye(3)))):
            return True
        vals, vecs = np.linalg.eigh(symm)
        # A mirror plane
        if np.allclose(np.abs(vals), 1) and np.allclose(np.prod(vals), -1):
            # Mapping k -> k, modulo a lattice vector
            if np.allclose(kpt % 1, (np.dot(symm, kpt)) % 1):
                mirror_count += 1
    # If we have two or more mirror planes, then we must have a spin-degenerate
    # subspace
    if mirror_count >= 2:
        return True
    return False


def eigenvalues(calc):
    """
    Parameters:
        calc: Calculator
            GPAW calculator
    Returns:
        e_skn: (ns, nk, nb)-shape array
    """
    import numpy as np
    rs = range(calc.get_number_of_spins())
    rk = range(len(calc.get_ibz_k_points()))
    e = calc.get_eigenvalues
    return np.asarray([[e(spin=s, kpt=k) for k in rk] for s in rs])


def get_spin_direction(fname='anisotropy_xy.npz'):
    '''
    Uses the magnetic anisotropy to calculate the preferred spin orientation
    for magnetic (FM/AFM) systems.

    Parameters:
        fname:
            The filename of a datafile containing the xz and yz
            anisotropy energies.
    Returns:
        theta:
            Polar angle in radians
        phi:
            Azimuthal angle in radians
    '''

    import numpy as np
    import os.path as op
    theta = 0
    phi = 0
    if op.isfile(fname):
        data = np.load(fname)
        DE = max(data['dE_zx'], data['dE_zy'])
        if DE > 0:
            theta = np.pi / 2
            if data['dE_zy'] > data['dE_zx']:
                phi = np.pi / 2
    return theta, phi


def spin_axis(fname='anisotropy_xy.npz') -> int:
    import numpy as np
    theta, phi = get_spin_direction(fname=fname)
    if theta == 0:
        return 2
    elif np.allclose(phi, np.pi / 2):
        return 1
    else:
        return 0


def collect_data(atoms):
    """Band structure PBE and GW +- SOC."""
    import os.path as op
    from pathlib import Path
    import json
    from ase.io import jsonio
    kvp = {}
    key_descriptions = {}
    data = {}

    if not op.isfile('results-bs-soc.json'):
        return kvp, key_descriptions, data

    import numpy as np
    evac = kvp.get('evac')
    soc = json.loads(jsonio.decode(Path('results-bs-soc.json').read_text()))
    nosoc = json.loads(jsonio.decode(
        Path('results-bs-nosoc.json').read_text()))
    eps_skn = nosoc['energies']
    path = soc['path']
    npoints = len(path['kpts'])
    s_mvk = np.array(soc.get('spin_mvk'))
    if s_mvk.ndim == 3:
        sz_mk = s_mvk[:, spin_axis(), :]  # take x, y or z component
    else:
        sz_mk = s_mvk

    assert sz_mk.shape[1] == npoints, f'sz_mk has wrong dims, {npoints}'

    efermi = np.load('gap_soc.npz')['efermi']
    efermi_nosoc = np.load('gap.npz')['efermi']

    pbe = {
        'path': path,
        'eps_skn': eps_skn,
        'efermi_nosoc': efermi_nosoc,
        'efermi': efermi,
        'eps_so_mk': soc['energies'],
        'sz_mk': sz_mk,
        'evac': evac or np.nan
    }
    try:
        op_scc = data['op_scc']
    except KeyError:
        from gpaw.symmetry import atoms2symmetry
        op_scc = atoms2symmetry(atoms).op_scc
    import json
    from pathlib import Path
    magstate = json.loads(Path('quickinfo.json').read_text())['magstate']

    for idx, kpt in enumerate(path):
        if (magstate == 'NM' and is_symmetry_protected(kpt, op_scc)
                or magstate == 'AFM'):
            pbe['sz_mk'][:, idx] = 0.0

    data['bs_pbe'] = pbe
    return kvp, key_descriptions, data


def bs_pbe_html(row,
                filename='pbe-bs.html',
                figsize=(6.4, 4.8),
                fontsize=10,
                show_legend=True,
                s=2):
    if 'bs_pbe' not in row.data or 'eps_so_mk' not in row.data.bs_pbe:
        return

    import plotly
    import plotly.graph_objs as go
    import numpy as np

    traces = []
    d = row.data.bs_pbe
    e_skn = d['eps_skn']
    path = d['path']
    kpts = path.kpts
    ef = d['efermi']
    emin = row.get('vbm', ef) - 5
    emax = row.get('cbm', ef) + 5
    shape = e_skn.shape

    from ase.dft.kpoints import labels_from_kpts
    xcoords, label_xcoords, orig_labels = labels_from_kpts(kpts, row.cell)
    xcoords = np.vstack([xcoords] * shape[0] * shape[2])
    # colors_s = plt.get_cmap('viridis')([0, 1])  # color for sz = 0
    e_kn = np.hstack([e_skn[x] for x in range(shape[0])])
    trace = go.Scattergl(
        x=xcoords.ravel(),
        y=e_kn.T.ravel(),
        mode='markers',
        name='PBE no SOC',
        showlegend=True,
        marker=dict(size=4, color='#999999'))
    traces.append(trace)

    d = row.data.bs_pbe
    e_mk = d['eps_so_mk']
    path = d['path']
    kpts = path.kpts
    ef = d['efermi']
    sz_mk = d['sz_mk']
    emin = row.get('vbm', ef) - 5
    emax = row.get('cbm', ef) + 5

    from ase.dft.kpoints import labels_from_kpts
    xcoords, label_xcoords, orig_labels = labels_from_kpts(kpts, row.cell)

    shape = e_mk.shape
    perm = (sz_mk).argsort(axis=None)
    e_mk = e_mk.ravel()[perm].reshape(shape)
    sz_mk = sz_mk.ravel()[perm].reshape(shape)
    xcoords = np.vstack([xcoords] * shape[0])
    xcoords = xcoords.ravel()[perm].reshape(shape)

    # Unicode for <S_z>
    sdir = row.get('spin_orientation', 'z')
    cbtitle = '&#x3008; <i><b>S</b></i><sub>{}</sub> &#x3009;'.format(sdir)
    trace = go.Scattergl(
        x=xcoords.ravel(),
        y=e_mk.ravel(),
        mode='markers',
        name='PBE',
        showlegend=True,
        marker=dict(
            size=4,
            color=sz_mk.ravel(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                tickmode='array',
                tickvals=[-1, 0, 1],
                ticktext=['-1', '0', '1'],
                title=cbtitle,
                titleside='right')))
    traces.append(trace)

    linetrace = go.Scatter(
        x=[np.min(xcoords), np.max(xcoords)],
        y=[ef, ef],
        mode='lines',
        line=dict(color=('rgb(0, 0, 0)'), width=2, dash='dash'),
        name='Fermi level')
    traces.append(linetrace)

    def pretty(kpt):
        if kpt == 'G':
            kpt = '&#x393;'  # Gamma in unicode
        elif len(kpt) == 2:
            kpt = kpt[0] + '$_' + kpt[1] + '$'
        return kpt

    labels = [pretty(name) for name in orig_labels]
    i = 1
    while i < len(labels):
        if label_xcoords[i - 1] == label_xcoords[i]:
            labels[i - 1] = labels[i - 1][:-1] + ',' + labels[i][1:]
            labels[i] = ''
        i += 1

    bandxaxis = go.layout.XAxis(
        title="k-points",
        range=[0, np.max(xcoords)],
        showgrid=True,
        showline=True,
        ticks="",
        showticklabels=True,
        mirror=True,
        linewidth=2,
        ticktext=labels,
        tickvals=label_xcoords,
    )

    bandyaxis = go.layout.YAxis(
        title="<i>E</i> [eV]",
        range=[emin, emax],
        showgrid=True,
        showline=True,
        zeroline=False,
        mirror="ticks",
        ticks="inside",
        linewidth=2,
        tickwidth=2,
        zerolinewidth=2,
    )

    bandlayout = go.Layout(
        xaxis=bandxaxis,
        yaxis=bandyaxis,
        legend=dict(x=0, y=1),
        hovermode='closest',
        margin=dict(t=40, r=100),
        font=dict(size=18))

    fig = {'data': traces, 'layout': bandlayout}
    # fig['layout']['margin'] = {'t': 40, 'r': 100}
    # fig['layout']['hovermode'] = 'closest'
    # fig['layout']['legend'] =

    plot_html = plotly.offline.plot(
        fig, include_plotlyjs=False, output_type='div')
    # plot_html = ''.join(['<div style="width: 1000px;',
    #                      'height=1000px;">',
    #                      plot_html,
    #                      '</div>'])

    inds = []
    for i, c in enumerate(plot_html):
        if c == '"':
            inds.append(i)
    plotdivid = plot_html[inds[0] + 1:inds[1]]

    resize_script = (
        ''
        '<script type="text/javascript">'
        'window.addEventListener("resize", function(){{'
        'Plotly.Plots.resize(document.getElementById("{id}"));}});'
        '</script>').format(id=plotdivid)

    # Insert plotly.js
    plotlyjs = ('<script src="https://cdn.plot.ly/plotly-latest.min.js">' +
                '</script>')

    html = ''.join([
        '<html>', '<head><meta charset="utf-8" /></head>', '<body>', plotlyjs,
        plot_html, resize_script, '</body>', '</html>'
    ])

    with open(filename, 'w') as fd:
        fd.write(html)


def add_bs_pbe(row, ax, **kwargs):
    """plot pbe with soc on ax"""
    from ase.dft.kpoints import labels_from_kpts
    c = '0.8'  # light grey for pbe with soc plot
    ls = '-'
    lw = kwargs.get('lw', 1.0)
    d = row.data.bs_pbe
    kpts = d['path']
    e_mk = d['eps_so_mk']
    xcoords, label_xcoords, labels = labels_from_kpts(kpts, row.cell)
    for e_k in e_mk[:-1]:
        ax.plot(xcoords, e_k, color=c, ls=ls, lw=lw, zorder=-2)
    ax.lines[-1].set_label('PBE')
    ef = d['efermi']
    ax.axhline(ef, ls=':', zorder=0, color=c, lw=lw)
    return ax


def plot_with_colors(bs,
                     ax=None,
                     emin=-10,
                     emax=5,
                     filename=None,
                     show=None,
                     energies=None,
                     colors=None,
                     ylabel=None,
                     clabel='$s_z$',
                     cmin=-1.0,
                     cmax=1.0,
                     sortcolors=False,
                     loc=None,
                     s=2):
    """Plot band-structure with colors."""
    import numpy as np
    import matplotlib.pyplot as plt

    # if bs.ax is None:
    #     ax = bs.prepare_plot(ax, emin, emax, ylabel)
    # trying to find vertical lines and putt them in the back

    def vlines2back(lines):
        zmin = min([l.get_zorder() for l in lines])
        for l in lines:
            x = l.get_xdata()
            if len(x) > 0 and np.allclose(x, x[0]):
                l.set_zorder(zmin - 1)

    vlines2back(ax.lines)
    shape = energies.shape
    xcoords = np.vstack([bs.xcoords] * shape[1])
    if sortcolors:
        perm = (-colors).argsort(axis=None)
        energies = energies.ravel()[perm].reshape(shape)
        colors = colors.ravel()[perm].reshape(shape)
        xcoords = xcoords.ravel()[perm].reshape(shape)

    for e_k, c_k, x_k in zip(energies, colors, xcoords):
        things = ax.scatter(x_k, e_k, c=c_k, s=s, vmin=cmin, vmax=cmax)

    cbar = plt.colorbar(things)
    cbar.set_label(clabel)

    bs.finish_plot(filename, show, loc)

    return ax, cbar


def bs_pbe(row,
           filename='pbe-bs.png',
           figsize=(6.4, 4.8),
           fontsize=10,
           show_legend=True,
           s=0.5):
    if 'bs_pbe' not in row.data or 'eps_so_mk' not in row.data.bs_pbe:
        return
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    import numpy as np
    from ase.dft.band_structure import BandStructure
    d = row.data.bs_pbe
    e_skn = d['eps_skn']
    nspins = e_skn.shape[0]
    e_kn = np.hstack([e_skn[x] for x in range(nspins)])[np.newaxis]
    path = d['path']
    ef = d['efermi']
    emin = row.get('vbm', ef) - 3 - ef
    emax = row.get('cbm', ef) + 3 - ef
    mpl.rcParams['font.size'] = fontsize
    bs = BandStructure(path, e_kn, ef)
    # pbe without soc
    nosoc_style = dict(
        colors=['0.8'] * e_skn.shape[0],
        label='PBE no SOC',
        ls='-',
        lw=1.0,
        zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    bsp = bs.plot(
        ax=ax,
        show=False,
        emin=emin,
        emax=emax,
        ylabel=r'$E$ [eV]',
        **nosoc_style)
    # pbe with soc
    e_mk = d['eps_so_mk']
    sz_mk = d['sz_mk']
    ax.figure.set_figheight(1.2 * ax.figure.get_figheight())
    sdir = row.get('spin_orientation', 'z')
    ax, cbar = plot_with_colors(
        bsp,
        ax=ax,
        energies=e_mk,
        colors=sz_mk,
        filename=filename,
        show=False,
        emin=emin,
        emax=emax,
        sortcolors=True,
        loc='upper right',
        clabel=r'$\langle S_{} \rangle $'.format(sdir),
        s=s)

    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.update_ticks()
    csz0 = plt.get_cmap('viridis')(0.5)  # color for sz = 0
    ax.plot([], [], label='PBE', color=csz0)
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


def webpanel(row, key_descriptions):
    from asr.custom import fig, table
    from typing import Tuple, List

    def rmxclabel(d: 'Tuple[str, str, str]',
                  xcs: List) -> 'Tuple[str, str, str]':
        def rm(s: str) -> str:
            for xc in xcs:
                s = s.replace('({})'.format(xc), '')
            return s.rstrip()

        return tuple(rm(s) for s in d)

    xcs = ['PBE', 'GLLBSC', 'HSE', 'GW']
    key_descriptions_noxc = {
        k: rmxclabel(d, xcs)
        for k, d in key_descriptions.items()
    }

    if row.get('gap', 0) > 0.0:
        if row.get('evacdiff', 0) > 0.02:
            pbe = table(
                row,
                'Property', [
                    'work_function', 'gap', 'dir_gap', 'vbm', 'cbm', 'D_vbm',
                    'D_cbm', 'dipz', 'evacdiff'
                ],
                kd=key_descriptions_noxc)
        else:
            pbe = table(
                row,
                'Property', [
                    'work_function', 'gap', 'dir_gap', 'vbm', 'cbm', 'D_vbm',
                    'D_cbm'
                ],
                kd=key_descriptions_noxc)
    else:
        if row.get('evacdiff', 0) > 0.02:
            pbe = table(
                row,
                'Property', [
                    'work_function', 'dosef_soc', 'gap', 'dir_gap', 'vbm',
                    'cbm', 'D_vbm', 'D_cbm', 'dipz', 'evacdiff'
                ],
                kd=key_descriptions_noxc)
        else:
            pbe = table(
                row,
                'Property', [
                    'work_function', 'dosef_soc', 'gap', 'dir_gap', 'vbm',
                    'cbm', 'D_vbm', 'D_cbm'
                ],
                kd=key_descriptions_noxc)

    panel = ('Electronic band structure (PBE)',
             [[fig('pbe-bs.png', link='pbe-bs.html'),
               fig('bz.png')], [fig('pbe-pdos.png', link='empty'), pbe]])

    things = [(bs_pbe, ['pbe-bs.png']), (bs_pbe_html, ['pbe-bs.html'])]
    return panel, things


group = 'Property'
creates = ['bs.gpw', 'results-bs-soc.json', 'results-bs-nosoc.json']
dependencies = ['asr.gs']

if __name__ == '__main__':
    main(standalone_mode=False)
