import functools
import warnings
import numpy as np
import typing
from typing import List, Tuple, Optional, Dict, Any
from ase.formula import Formula
from ase.db.row import AtomsRow
from ase.phasediagram import PhaseDiagram
from ase.dft.kpoints import labels_from_kpts
from asr.core import ASRResult, prepare_result
from asr.database.browser import (
    WebPanel,
    create_table, table,
    fig,
    href, dl, code,
    describe_entry,
    entry_parameter_description,
    make_panel_description)
from asr.utils.hacks import gs_xcname_from_row
from matplotlib import patches


######### Optical Panel #########
def OpticalWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The frequency-dependent polarisability in the long wave length limit (q=0)
    calculated in the random phase approximation (RPA) without spin–orbit
    interactions. For metals a Drude term accounts for intraband transitions. The
    contribution from polar lattice vibrations is added (see infrared
    polarisability) and may be visible at low frequencies.""",
        articles=['C2DB'])

    explanation = 'Optical polarizability along the'
    alphax_el = describe_entry('alphax_el',
                               description=explanation + " x-direction")
    alphay_el = describe_entry('alphay_el',
                               description=explanation + " y-direction")
    alphaz_el = describe_entry('alphaz_el',
                               description=explanation + " z-direction")

    opt = create_table(row=row, header=['Property', 'Value'],
                       keys=[alphax_el, alphay_el, alphaz_el],
                       key_descriptions=key_descriptions, digits=2)

    panel = {'title': describe_entry('Optical polarizability',
                                     panel_description),
             'columns': [[fig('rpa-pol-x.png'), fig('rpa-pol-z.png')],
                         [fig('rpa-pol-y.png'), opt]],
             'plot_descriptions':
                 [{'function': polarizability,
                   'filenames': ['rpa-pol-x.png',
                                 'rpa-pol-y.png',
                                 'rpa-pol-z.png']}],
             'subpanel': 'Polarizabilities',
             'sort': 20}

    return [panel]

@prepare_result
class OpticalResult(ASRResult):
    alphax_el: typing.List[complex]
    alphay_el: typing.List[complex]
    alphaz_el: typing.List[complex]
    plasmafreq_vv: typing.List[typing.List[float]]
    frequencies: typing.List[float]

    key_descriptions = {
        "alphax_el": "Optical polarizability (x) [Ang]",
        "alphay_el": "Optical polarizability (y) [Ang]",
        "alphaz_el": "Optical polarizability (z) [Ang]",
        "plasmafreq_vv": "Plasmafrequency tensor.",
        "frequencies": "Frequency grid [eV]."
    }

    formats = {"ase_webpanel": OpticalWebpanel}


def polarizability(row, fx, fy, fz):
    import matplotlib.pyplot as plt

    def ylims(ws, data, wstart=0.0):
        i = abs(ws - wstart).argmin()
        x = data[i:]
        x1, x2 = x.real, x.imag
        y1 = min(x1.min(), x2.min()) * 1.02
        y2 = max(x1.max(), x2.max()) * 1.02
        return y1, y2

    def plot_polarizability(ax, frequencies, alpha_w, filename, direction):
        ax.set_title(f'Polarization: {direction}')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel(r'Polarizability [$\mathrm{\AA}$]')
        ax.set_ylim(ylims(ws=frequencies, data=alpha_w, wstart=0.5))
        ax.legend()
        ax.set_xlim((0, 10))
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(filename)

    data = row.data.get('results-asr.polarizability.json')

    if data is None:
        return
    frequencies = data['frequencies']
    i2 = abs(frequencies - 50.0).argmin()
    frequencies = frequencies[:i2]
    alphax_w = data['alphax_w'][:i2]
    alphay_w = data['alphay_w'][:i2]
    alphaz_w = data['alphaz_w'][:i2]

    ax = plt.figure().add_subplot(111)
    ax1 = ax
    try:
        wpx = row.plasmafrequency_x
        if wpx > 0.01:
            alphaxfull_w = alphax_w - wpx**2 / (2 * np.pi * (frequencies + 1e-9)**2)
            ax.plot(
                frequencies,
                np.real(alphaxfull_w),
                '-',
                c='C1',
                label='real')
            ax.plot(
                frequencies,
                np.real(alphax_w),
                '--',
                c='C1',
                label='real (interband)')
        else:
            ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
    except AttributeError:
        ax.plot(frequencies, np.real(alphax_w), c='C1', label='real')
    ax.plot(frequencies, np.imag(alphax_w), c='C0', label='imag')

    plot_polarizability(ax, frequencies, alphax_w, filename=fx, direction='x')

    ax = plt.figure().add_subplot(111)
    ax2 = ax
    try:
        wpy = row.plasmafrequency_y
        if wpy > 0.01:
            alphayfull_w = alphay_w - wpy**2 / (2 * np.pi * (frequencies + 1e-9)**2)
            ax.plot(
                frequencies,
                np.real(alphayfull_w),
                '-',
                c='C1',
                label='real')
            ax.plot(
                frequencies,
                np.real(alphay_w),
                '--',
                c='C1',
                label='real (interband)')
        else:
            ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')
    except AttributeError:
        ax.plot(frequencies, np.real(alphay_w), c='C1', label='real')

    ax.plot(frequencies, np.imag(alphay_w), c='C0', label='imag')
    plot_polarizability(ax, frequencies, alphay_w, filename=fy, direction='y')

    ax3 = plt.figure().add_subplot(111)
    ax3.plot(frequencies, np.real(alphaz_w), c='C1', label='real')
    ax3.plot(frequencies, np.imag(alphaz_w), c='C0', label='imag')
    plot_polarizability(ax3, frequencies, alphaz_w, filename=fz, direction='z')

    return ax1, ax2, ax3


######### Plasma Panel #########
def PlasmaWebpanel(result, row, key_descriptions):
    from asr.database.browser import table

    if row.get('gap', 1) > 0.01:
        return []

    plasmatable = table(row, 'Property', [
        'plasmafrequency_x', 'plasmafrequency_y'], key_descriptions)

    panel = {'title': 'Optical polarizability',
             'columns': [[], [plasmatable]]}
    return [panel]


@prepare_result
class PlasmaResult(ASRResult):

    plasmafreq_vv: typing.List[typing.List[float]]
    plasmafrequency_x: float
    plasmafrequency_y: float

    key_descriptions = {
        "plasmafreq_vv": "Plasma frequency tensor [Hartree]",
        "plasmafrequency_x": "KVP: 2D plasma frequency (x)"
        "[`eV/Å^0.5`]",
        "plasmafrequency_y": "KVP: 2D plasma frequency (y)"
        "[`eV/Å^0.5`]",
    }
    formats = {"ase_webpanel": PlasmaWebpanel}


######### Infrared Panel #########
def InfraredWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The frequency-dependent polarisability in the infrared (IR) frequency regime
    calculated from a Lorentz oscillator equation involving the optical Gamma-point
    phonons and atomic Born charges. The contribution from electronic interband
    transitions is added, but is essentially constant for frequencies much smaller
    than the direct band gap.
    """,
        articles=[
            href("""\
    M. N. Gjerding et al. Efficient Ab Initio Modeling of Dielectric Screening
    in 2D van der Waals Materials: Including Phonons, Substrates, and Doping,
    J. Phys. Chem. C 124 11609 (2020)""",
                 'https://doi.org/10.1021/acs.jpcc.0c01635'),
        ])
    explanation = 'Static lattice polarizability along the'
    alphax_lat = describe_entry('alphax_lat', description=explanation + " x-direction")
    alphay_lat = describe_entry('alphay_lat', description=explanation + " y-direction")
    alphaz_lat = describe_entry('alphaz_lat', description=explanation + " z-direction")

    explanation = 'Total static polarizability along the'
    alphax = describe_entry('alphax', description=explanation + " x-direction")
    alphay = describe_entry('alphay', description=explanation + " y-direction")
    alphaz = describe_entry('alphaz', description=explanation + " z-direction")

    opt = table(
        row, "Property", [alphax_lat, alphay_lat, alphaz_lat, alphax,
                          alphay, alphaz], key_descriptions
    )

    panel = {
        "title": describe_entry("Infrared polarizability",
                                panel_description),
        "columns": [[fig("infrax.png"), fig("infraz.png")], [fig("infray.png"), opt]],
        "plot_descriptions": [
            {
                "function": create_plot,
                "filenames": ["infrax.png", "infray.png", "infraz.png"],
            }
        ],
        "subpanel": 'Polarizabilities',
        "sort": 21,
    }

    return [panel]


def create_plot(row, *fnames):
    infrareddct = row.data['results-asr.infraredpolarizability.json']
    electrondct = row.data['results-asr.polarizability.json']
    phonondata = row.data['results-asr.phonons.json']
    maxphononfreq = phonondata['omega_kl'][0].max() * 1e3

    assert len(fnames) == 3
    for v, (axisname, fname) in enumerate(zip('xyz', fnames)):
        alpha_w = electrondct[f'alpha{axisname}_w']

        create_plot_simple(
            ndim=sum(row.toatoms().pbc),
            maxomega=maxphononfreq * 1.5,
            omega_w=infrareddct["omega_w"] * 1e3,
            alpha_w=alpha_w,
            alphavv_w=infrareddct["alpha_wvv"][:, v, v],
            omegatmp_w=electrondct["frequencies"] * 1e3,
            axisname=axisname,
            fname=fname)


def create_plot_simple(*, ndim, omega_w, fname, maxomega, alpha_w,
                       alphavv_w, axisname,
                       omegatmp_w):
    from scipy.interpolate import interp1d

    re_alpha = interp1d(omegatmp_w, alpha_w.real)
    im_alpha = interp1d(omegatmp_w, alpha_w.imag)
    a_w = (re_alpha(omega_w) + 1j * im_alpha(omega_w) + alphavv_w)

    if ndim == 3:
        ylabel = r'Dielectric function'
        yvalues = 1 + 4 * np.pi * a_w
    else:
        power_txt = {2: '', 1: '^2', 0: '^3'}[ndim]
        unit = rf"$\mathrm{{\AA}}{power_txt}$"
        ylabel = rf'Polarizability [{unit}]'
        yvalues = a_w

    return mkplot(yvalues, axisname, fname, maxomega, omega_w, ylabel)


def mkplot(a_w, axisname, fname, maxomega, omega_w, ylabel):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(omega_w, a_w.real, c='C1', label='real')
    ax.plot(omega_w, a_w.imag, c='C0', label='imag')
    ax.set_title(f'Polarization: {axisname}')
    ax.set_xlabel('Energy [meV]')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, maxomega)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname)
    return fname


@prepare_result
class InfraredResult(ASRResult):

    alpha_wvv: typing.List[typing.List[typing.List[complex]]]
    omega_w: typing.List[float]
    alphax_lat: complex
    alphay_lat: complex
    alphaz_lat: complex
    alphax: complex
    alphay: complex
    alphaz: complex

    key_descriptions = {
        "alpha_wvv": "Lattice polarizability.",
        "omega_w": "Frequency grid [eV].",
        "alphax_lat": "Lattice polarizability at omega=0 (x-direction).",
        "alphay_lat": "Lattice polarizability at omega=0 (y-direction).",
        "alphaz_lat": "Lattice polarizability at omega=0 (z-direction).",
        "alphax": "Lattice+electronic polarizability at omega=0 (x-direction).",
        "alphay": "Lattice+electronic polarizability at omega=0 (y-direction).",
        "alphaz": "Lattice+electronic polarizability at omega=0 (z-direction).",
    }

    formats = {"ase_webpanel": InfraredWebpanel}


######### Bader #########
def BaderWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The Bader charge analysis ascribes a net charge to an atom
    by partitioning the electron density according to its zero-flux surfaces.""",
        articles=[
            href("""W. Tang et al. A grid-based Bader analysis algorithm
    without lattice bias. J. Phys.: Condens. Matter 21, 084204 (2009).""",
                 'https://doi.org/10.1088/0953-8984/21/8/084204')])
    rows = [[str(a), symbol, f'{charge:.2f}']
            for a, (symbol, charge)
            in enumerate(zip(result.sym_a, result.bader_charges))]
    table = {'type': 'table',
             'header': ['Atom index', 'Atom type', 'Charge (|e|)'],
             'rows': rows}

    parameter_description = entry_parameter_description(
        row.data,
        'asr.bader')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Bader charges',
                                     description=title_description),
             'columns': [[table]]}

    return [panel]


@prepare_result
class BaderResult(ASRResult):

    bader_charges: np.ndarray
    sym_a: List[str]

    key_descriptions = {'bader_charges': 'Array of charges [\\|e\\|].',
                        'sym_a': 'Chemical symbols.'}

    formats = {"ase_webpanel": BaderWebpanel}


######### Bandstructure #########
bs_png = 'bs.png'
bs_html = 'bs.html'


def plot_bs_html(row,
                 filename=bs_html,
                 figsize=(6.4, 6.4),
                 s=2):
    import plotly
    import plotly.graph_objs as go

    traces = []
    d = row.data.get('results-asr.bandstructure.json')
    xcname = gs_xcname_from_row(row)

    path = d['bs_nosoc']['path']
    kpts = path.kpts
    ef = d['bs_nosoc']['efermi']

    if row.get('evac') is not None:
        label = '<i>E</i> - <i>E</i><sub>vac</sub> [eV]'
        reference = row.get('evac')
    else:
        label = '<i>E</i> - <i>E</i><sub>F</sub> [eV]'
        reference = ef

    gaps = row.data.get('results-asr.gs.json', {}).get('gaps_nosoc', {})
    if gaps.get('vbm'):
        emin = gaps.get('vbm', ef) - 3
    else:
        emin = ef - 3
    if gaps.get('cbm'):
        emax = gaps.get('cbm', ef) + 3
    else:
        emax = ef + 3
    e_skn = d['bs_nosoc']['energies']
    shape = e_skn.shape
    xcoords, label_xcoords, orig_labels = labels_from_kpts(kpts, row.cell)
    xcoords = np.vstack([xcoords] * shape[0] * shape[2])
    # colors_s = plt.get_cmap('viridis')([0, 1])  # color for sz = 0
    e_kn = np.hstack([e_skn[x] for x in range(shape[0])])
    trace = go.Scattergl(
        x=xcoords.ravel(),
        y=e_kn.T.ravel() - reference,
        mode='markers',
        name=f'{xcname} no SOC',
        showlegend=True,
        marker=dict(size=4, color='#999999'))
    traces.append(trace)

    e_mk = d['bs_soc']['energies']
    path = d['bs_soc']['path']
    kpts = path.kpts
    ef = d['bs_soc']['efermi']
    sz_mk = d['bs_soc']['sz_mk']

    xcoords, label_xcoords, orig_labels = labels_from_kpts(kpts, row.cell)

    shape = e_mk.shape
    perm = (-sz_mk).argsort(axis=None)
    e_mk = e_mk.ravel()[perm].reshape(shape)
    sz_mk = sz_mk.ravel()[perm].reshape(shape)
    xcoords = np.vstack([xcoords] * shape[0])
    xcoords = xcoords.ravel()[perm].reshape(shape)

    # Unicode for <S_z>
    sdir = row.get('spin_axis', 'z')
    cbtitle = '&#x3008; <i><b>S</b></i><sub>{}</sub> &#x3009;'.format(sdir)
    trace = go.Scattergl(
        x=xcoords.ravel(),
        y=e_mk.ravel() - reference,
        mode='markers',
        name=xcname,
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
        y=[ef - reference, ef - reference],
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
        title=label,
        range=[emin - reference, emax - reference],
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
    plotlyjs = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

    html = ''.join([
        '<html>', '<head><meta charset="utf-8" /></head>', '<body>', plotlyjs,
        plot_html, resize_script, '</body>', '</html>'
    ])

    with open(filename, 'w') as fd:
        fd.write(html)


def add_bs_ks(row, ax, reference=0, color='C1'):
    """Plot with soc on ax."""
    d = row.data.get('results-asr.bandstructure.json')
    path = d['bs_soc']['path']
    e_mk = d['bs_soc']['energies']
    xcname = gs_xcname_from_row(row)
    xcoords, label_xcoords, labels = labels_from_kpts(path.kpts, row.cell)
    for e_k in e_mk[:-1]:
        ax.plot(xcoords, e_k - reference, color=color, zorder=-2)
    ax.lines[-1].set_label(xcname)
    ef = d['bs_soc']['efermi']
    ax.axhline(ef - reference, ls=':', zorder=-2, color=color)
    return ax


def plot_with_colors(bs,
                     ax=None,
                     emin=-10,
                     emax=5,
                     filename=None,
                     show=None,
                     energies=None,
                     colors=None,
                     colorbar=True,
                     ylabel=None,
                     clabel='$s_z$',
                     cmin=-1.0,
                     cmax=1.0,
                     sortcolors=False,
                     loc=None,
                     s=2):
    """Plot band-structure with colors."""
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
    xcoords = np.vstack([bs.xcoords] * shape[0])
    if sortcolors:
        perm = (-colors).argsort(axis=None)
        energies = energies.ravel()[perm].reshape(shape)
        colors = colors.ravel()[perm].reshape(shape)
        xcoords = xcoords.ravel()[perm].reshape(shape)

    for e_k, c_k, x_k in zip(energies, colors, xcoords):
        things = ax.scatter(x_k, e_k, c=c_k, s=s, vmin=cmin, vmax=cmax)

    if colorbar:
        cbar = plt.colorbar(things)
        cbar.set_label(clabel)
    else:
        cbar = None

    bs.finish_plot(filename, show, loc)

    return ax, cbar


def legend_on_top(ax, **kwargs):
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1, 1, 0),
              mode='expand', **kwargs)


def plot_bs_png(row,
                filename=bs_png,
                figsize=(5.5, 5),
                s=0.5):

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.patheffects as path_effects
    from ase.spectrum.band_structure import BandStructure, BandStructurePlot
    d = row.data.get('results-asr.bandstructure.json')
    xcname = gs_xcname_from_row(row)

    path = d['bs_nosoc']['path']
    ef_nosoc = d['bs_nosoc']['efermi']
    ef_soc = d['bs_soc']['efermi']
    ref_nosoc = row.get('evac', d.get('bs_nosoc').get('efermi'))
    ref_soc = row.get('evac', d.get('bs_soc').get('efermi'))
    if row.get('evac') is not None:
        label = r'$E - E_\mathrm{vac}$ [eV]'
    else:
        label = r'$E - E_\mathrm{F}$ [eV]'

    e_skn = d['bs_nosoc']['energies']
    nspins = e_skn.shape[0]
    e_kn = np.hstack([e_skn[x] for x in range(nspins)])[np.newaxis]

    gaps = row.data.get('results-asr.gs.json', {}).get('gaps_nosoc', {})
    if gaps.get('vbm'):
        emin = gaps.get('vbm') - 3
    else:
        emin = ef_nosoc - 3
    if gaps.get('cbm'):
        emax = gaps.get('cbm') + 3
    else:
        emax = ef_nosoc + 3
    bs = BandStructure(path, e_kn - ref_nosoc, ef_soc - ref_soc)
    # without soc
    nosoc_style = dict(
        colors=['0.8'] * e_skn.shape[0],
        label=f'{xcname} no SOC',
        ls='-',
        lw=1.0,
        zorder=0)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    bsp = BandStructurePlot(bs)
    bsp.plot(
        ax=ax,
        show=False,
        emin=emin - ref_nosoc,
        emax=emax - ref_nosoc,
        ylabel=label,
        **nosoc_style)
    # with soc
    e_mk = d['bs_soc']['energies']
    sz_mk = d['bs_soc']['sz_mk']
    sdir = row.get('spin_axis', 'z')
    colorbar = not (row.magstate == 'NM' and row.has_inversion_symmetry)
    ax, cbar = plot_with_colors(
        bsp,
        ax=ax,
        energies=e_mk - ref_soc,
        colors=sz_mk,
        colorbar=colorbar,
        filename=filename,
        show=False,
        emin=emin - ref_soc,
        emax=emax - ref_soc,
        sortcolors=True,
        loc='upper right',
        clabel=r'$\langle S_{} \rangle $'.format(sdir),
        s=s)

    if cbar:
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.update_ticks()
    csz0 = plt.get_cmap('viridis')(0.5)  # color for sz = 0
    ax.plot([], [], label=xcname, color=csz0)

    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    text = ax.annotate(
        r'$E_\mathrm{F}$',
        xy=(x0, ef_soc - ref_soc),
        fontsize=rcParams['font.size'] * 1.25,
        ha='left',
        va='bottom')

    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])
    legend_on_top(ax, ncol=2)
    plt.savefig(filename, bbox_inches='tight')


def BandstructureWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The band structure with spin–orbit interactions is shown with the
    expectation value of S_i (where i=z for non-magnetic materials and otherwise is
    the magnetic easy axis) indicated by the color code.""",
        articles=['C2DB'],
    )

    from typing import Tuple, List
    from asr.utils.hacks import gs_xcname_from_row

    def rmxclabel(d: 'Tuple[str, str, str]',
                  xcs: List) -> 'Tuple[str, str, str]':
        def rm(s: str) -> str:
            for xc in xcs:
                s = s.replace('({})'.format(xc), '')
            return s.rstrip()

        return tuple(rm(s) for s in d)

    xcname = gs_xcname_from_row(row)

    panel = {'title': describe_entry(f'Electronic band structure ({xcname})',
                                     panel_description),
             'columns': [
                 [
                     fig(bs_png, link=bs_html),
                 ],
                 [fig('bz-with-gaps.png')]],
             'plot_descriptions': [{'function': plot_bs_png,
                                    'filenames': [bs_png]},
                                   {'function': plot_bs_html,
                                    'filenames': [bs_html]}],
             'sort': 12}

    return [panel]


@prepare_result
class BandStructureResult(ASRResult):

    version: int = 0

    bs_soc: dict
    bs_nosoc: dict

    key_descriptions = \
        {
            'bs_soc': 'Bandstructure data with spin–orbit coupling.',
            'bs_nosoc': 'Bandstructure data without spin–orbit coupling.'
        }

    formats = {"ase_webpanel": BandstructureWebpanel}


######### Berry #########
olsen_title = ('T. Olsen et al. Discovering two-dimensional topological '
               'insulators from high-throughput computations. '
               'Phys. Rev. Mater. 3 024005.')
olsen_doi = 'https://doi.org/10.1103/PhysRevMaterials.3.024005'

panel_description = make_panel_description(
    """\
The spectrum was calculated by diagonalizing the Berry phase matrix
obtained by parallel transporting the occupied Bloch states along the
k₀-direction for each value of k₁. The eigenvalues can be interpreted
as the charge centers of hybrid Wannier functions localised in the
0-direction and the colours show the expectation values of spin for
the corresponding Wannier functions. A gapless spectrum is a minimal
requirement for non-trivial topological invariants.
""",
    articles=[href(olsen_title, olsen_doi)],
)


@prepare_result
class CalculateResult(ASRResult):

    phi0_km: np.ndarray
    phi1_km: np.ndarray
    phi2_km: np.ndarray
    phi0_pi_km: np.ndarray
    s0_km: np.ndarray
    s1_km: np.ndarray
    s2_km: np.ndarray
    s0_pi_km: np.ndarray

    key_descriptions = {
        'phi0_km': ('Berry phase spectrum at k_2=0, '
                    'localized along the k_0 direction'),
        'phi1_km': ('Berry phase spectrum at k_0=0, '
                    'localized along the k_1 direction'),
        'phi2_km': ('Berry phase spectrum at k_1=0, '
                    'localized along the k_2 direction'),
        'phi0_pi_km': ('Berry phase spectrum at k_2=pi, '
                       'localized along the k_0 direction'),
        's0_km': ('Expectation value of spin in the easy-axis direction '
                  'for the Berry phases at k_2=0'),
        's1_km': ('Expectation value of spin in the easy-axis direction '
                  'for the Berry phases at k_0=0'),
        's2_km': ('Expectation value of spin in the easy-axis direction '
                  'for the Berry phases at k_1=0'),
        's0_pi_km': ('Expectation value of spin in the easy-axis direction '
                     'for the Berry phases at k_2=pi'),
    }


def plot_phases(row, f0, f1, f2, fpi):
    import pylab as plt

    results = row.data['results-asr.berry@calculate.json']
    for f, label in [(f0, 0), (f1, 1), (f2, 2), (fpi, '0_pi')]:
        phit_km = results.get(f'phi{label}_km')
        if phit_km is None:
            continue
        St_km = results.get(f's{label}_km')
        if St_km is None:
            continue
        Nk = len(St_km)

        phi_km = np.zeros((len(phit_km) + 1, len(phit_km[0])), float)
        phi_km[1:] = phit_km
        phi_km[0] = phit_km[-1]
        S_km = np.zeros((len(phit_km) + 1, len(phit_km[0])), float)
        S_km[1:] = St_km
        S_km[0] = St_km[-1]
        S_km /= 2

        Nm = len(phi_km[0])
        phi_km = np.tile(phi_km, (1, 2))
        phi_km[:, Nm:] += 2 * np.pi
        S_km = np.tile(S_km, (1, 2))
        Nk = len(S_km)
        Nm = len(phi_km[0])

        shape = S_km.T.shape
        perm = np.argsort(S_km.T, axis=None)
        phi_km = phi_km.T.ravel()[perm].reshape(shape).T
        S_km = S_km.T.ravel()[perm].reshape(shape).T

        plt.figure()
        plt.scatter(np.tile(np.arange(Nk), Nm)[perm],
                    phi_km.T.reshape(-1),
                    cmap=plt.get_cmap('viridis'),
                    c=S_km.T.reshape(-1),
                    s=5,
                    marker='o')

        if 'results-asr.magnetic_anisotropy.json' in row.data:
            anis = row.data['results-asr.magnetic_anisotropy.json']
            dir = anis['spin_axis']
        else:
            dir = 'z'

        cbar = plt.colorbar()
        cbar.set_label(rf'$\langle S_{dir}\rangle/\hbar$', size=16)

        if f == f0:
            plt.title(r'$\tilde k_2=0$', size=22)
            plt.xlabel(r'$\tilde k_1$', size=20)
            plt.ylabel(r'$\gamma_0$', size=20)
        elif f == f1:
            plt.title(r'$\tilde k_0=0$', size=22)
            plt.xlabel(r'$\tilde k_2$', size=20)
            plt.ylabel(r'$\gamma_1$', size=20)
        if f == f2:
            plt.title(r'$\tilde k_1=0$', size=22)
            plt.xlabel(r'$\tilde k_0$', size=20)
            plt.ylabel(r'$\gamma_2$', size=20)
        if f == fpi:
            plt.title(r'$\tilde k_2=\pi$', size=22)
            plt.xlabel(r'$\tilde k_1$', size=20)
            plt.ylabel(r'$\gamma_0$', size=20)
        plt.xticks([0, Nk / 2, Nk],
                   [r'$-\pi$', r'$0$', r'$\pi$'], size=16)
        plt.yticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], size=16)
        plt.axis([0, Nk, 0, 2 * np.pi])
        plt.tight_layout()
        plt.savefig(f)


def BerryWebpanel(result, row, key_descriptions):
    from asr.utils.hacks import gs_xcname_from_row

    xcname = gs_xcname_from_row(row)
    parameter_description = entry_parameter_description(
        row.data,
        'asr.gs@calculate')
    description = ('Topological invariant characterizing the occupied bands \n\n'
                   + parameter_description)
    datarow = [describe_entry('Band topology', description), result.Topology]

    summary = WebPanel(title='Summary',
                       columns=[[{'type': 'table',
                                  'header': ['Basic properties', ''],
                                  'rows': [datarow]}]])

    basicelec = WebPanel(title=f'Basic electronic properties ({xcname})',
                         columns=[[{'type': 'table',
                                    'header': ['Property', ''],
                                    'rows': [datarow]}]],
                         sort=15)

    berry_phases = WebPanel(
        title=describe_entry('Berry phase', panel_description),
        columns=[[fig('berry-phases0.png'),
                  fig('berry-phases0_pi.png')],
                 [fig('berry-phases1.png'),
                  fig('berry-phases2.png')]],
        plot_descriptions=[{'function': plot_phases,
                            'filenames': ['berry-phases0.png',
                                          'berry-phases1.png',
                                          'berry-phases2.png',
                                          'berry-phases0_pi.png']}])

    return [summary, basicelec, berry_phases]


@prepare_result
class BerryResult(ASRResult):

    Topology: str

    key_descriptions = {'Topology': 'Band topology.'}
    formats = {"ase_webpanel": BerryWebpanel}

######### born charges #########
reference = """\
M. N. Gjerding et al. Efficient Ab Initio Modeling of Dielectric Screening
in 2D van der Waals Materials: Including Phonons, Substrates, and Doping,
J. Phys. Chem. C 124 11609 (2020)"""


def BornChargesWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The Born charge of an atom is defined as the derivative of the static
    macroscopic polarization w.r.t. its displacements u_i (i=x,y,z). The
    polarization in a periodic direction is calculated as an integral over Berry
    phases. The polarization in a non-periodic direction is obtained by direct
    evaluation of the first moment of the electron density. The Born charge is
    obtained as a finite difference of the polarization for displaced atomic
    configurations.  """,
        articles=[
            href(reference, 'https://doi.org/10.1021/acs.jpcc.0c01635')
        ]
    )
    import numpy as np

    def matrixtable(M, digits=2, unit='', skiprow=0, skipcolumn=0):
        table = M.tolist()
        shape = M.shape

        for i in range(skiprow, shape[0]):
            for j in range(skipcolumn, shape[1]):
                value = table[i][j]
                table[i][j] = '{:.{}f}{}'.format(value, digits, unit)
        return table

    columns = [[], []]
    for a, Z_vv in enumerate(
            row.data['results-asr.borncharges.json']['Z_avv']):
        table = np.zeros((4, 4))
        table[1:, 1:] = Z_vv
        rows = matrixtable(table, skiprow=1, skipcolumn=1)
        sym = row.symbols[a]
        rows[0] = [f'Z<sup>{sym}</sup><sub>ij</sub>', 'u<sub>x</sub>',
                   'u<sub>y</sub>', 'u<sub>z</sub>']
        rows[1][0] = 'P<sub>x</sub>'
        rows[2][0] = 'P<sub>y</sub>'
        rows[3][0] = 'P<sub>z</sub>'

        for ir, tmprow in enumerate(rows):
            for ic, item in enumerate(tmprow):
                if ir == 0 or ic == 0:
                    rows[ir][ic] = '<b>' + rows[ir][ic] + '</b>'

        Ztable = dict(
            type='table',
            rows=rows)

        columns[a % 2].append(Ztable)

    panel = {'title': describe_entry('Born charges', panel_description),
             'columns': columns,
             'sort': 17}
    return [panel]


@prepare_result
class BornChargesResult(ASRResult):

    Z_avv: np.ndarray
    sym_a: typing.List[str]

    key_descriptions = {'Z_avv': 'Array of borncharges.',
                        'sym_a': 'Chemical symbols.'}

    formats = {"ase_webpanel": BornChargesWebpanel}


######### bse #########
def absorption(row, filename, direction='x'):
    delta_bse, delta_rpa = gaps_from_row(row)
    return _absorption(
        dim=sum(row.toatoms().pbc),
        magstate=row.magstate,
        gap_dir=row.gap_dir,
        gap_dir_nosoc=row.gap_dir_nosoc,
        bse_data=np.array(
            row.data['results-asr.bse.json'][f'bse_alpha{direction}_w']),
        pol_data=row.data.get('results-asr.polarizability.json'),
        delta_bse=delta_bse,
        delta_rpa=delta_rpa,
        direction=direction,
        filename=filename)


def gaps_from_row(row):
    for method in ['_gw', '_hse', '_gllbsc', '']:
        gapkey = f'gap_dir{method}'
        if gapkey in row:
            gap_dir_x = row[gapkey]
            delta_bse = gap_dir_x - row.gap_dir
            delta_rpa = gap_dir_x - row.gap_dir_nosoc
            return delta_bse, delta_rpa


def _absorption(*, dim, magstate, gap_dir, gap_dir_nosoc,
                bse_data, pol_data,
                delta_bse, delta_rpa, filename, direction):
    import matplotlib.pyplot as plt
    from ase.units import alpha, Ha, Bohr

    qp_gap = gap_dir + delta_bse

    if magstate != 'NM':
        qp_gap = gap_dir_nosoc + delta_rpa
        delta_bse = delta_rpa

    ax = plt.figure().add_subplot(111)

    wbse_w = bse_data[:, 0] + delta_bse
    if dim == 2:
        sigma_w = -1j * 4 * np.pi * (bse_data[:, 1] + 1j * bse_data[:, 2])
        sigma_w *= wbse_w * alpha / Ha / Bohr
        absbse_w = np.real(sigma_w) * np.abs(2 / (2 + sigma_w))**2 * 100
    else:
        absbse_w = 4 * np.pi * bse_data[:, 2]
    ax.plot(wbse_w, absbse_w, '-', c='0.0', label='BSE')
    xmax = wbse_w[-1]

    # TODO: Sometimes RPA pol doesn't exist, what to do?
    if pol_data:
        wrpa_w = pol_data['frequencies'] + delta_rpa
        wrpa_w = pol_data['frequencies'] + delta_rpa
        if dim == 2:
            sigma_w = -1j * 4 * np.pi * pol_data[f'alpha{direction}_w']
            sigma_w *= wrpa_w * alpha / Ha / Bohr
            absrpa_w = np.real(sigma_w) * np.abs(2 / (2 + sigma_w))**2 * 100
        else:
            absrpa_w = 4 * np.pi * np.imag(pol_data[f'alpha{direction}_w'])
        ax.plot(wrpa_w, absrpa_w, '-', c='C0', label='RPA')
        ymax = max(np.concatenate([absbse_w[wbse_w < xmax],
                                   absrpa_w[wrpa_w < xmax]])) * 1.05
    else:
        ymax = max(absbse_w[wbse_w < xmax]) * 1.05

    ax.plot([qp_gap, qp_gap], [0, ymax], '--', c='0.5',
            label='Direct QP gap')

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, ymax)
    ax.set_title(f'Polarization: {direction}')
    ax.set_xlabel('Energy [eV]')
    if dim == 2:
        ax.set_ylabel('Absorbance [%]')
    else:
        ax.set_ylabel(r'$\varepsilon(\omega)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)

    return ax


def BSEWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """The optical absorption calculated from the Bethe–Salpeter Equation
    (BSE). The BSE two-particle Hamiltonian is constructed using the wave functions
    from a DFT calculation with the direct band gap adjusted to match the direct
    band gap from a G0W0 calculation. Spin–orbit interactions are included.  The
    result of the random phase approximation (RPA) with the same direct band gap
    adjustment as used for BSE but without spin–orbit interactions, is also shown.
    """,
        articles=['C2DB'],
    )

    import numpy as np
    from functools import partial

    E_B = table(row, 'Property', ['E_B'], key_descriptions)

    atoms = row.toatoms()
    pbc = atoms.pbc.tolist()
    dim = np.sum(pbc)

    if dim == 2:
        funcx = partial(absorption, direction='x')
        funcz = partial(absorption, direction='z')

        panel = {'title': describe_entry('Optical absorption (BSE and RPA)',
                                         panel_description),
                 'columns': [[fig('absx.png'), E_B],
                             [fig('absz.png')]],
                 'plot_descriptions': [{'function': funcx,
                                        'filenames': ['absx.png']},
                                       {'function': funcz,
                                        'filenames': ['absz.png']}]}
    else:
        funcx = partial(absorption, direction='x')
        funcy = partial(absorption, direction='y')
        funcz = partial(absorption, direction='z')

        panel = {'title': 'Optical absorption (BSE and RPA)',
                 'columns': [[fig('absx.png'), fig('absz.png')],
                             [fig('absy.png'), E_B]],
                 'plot_descriptions': [{'function': funcx,
                                        'filenames': ['absx.png']},
                                       {'function': funcy,
                                        'filenames': ['absy.png']},
                                       {'function': funcz,
                                        'filenames': ['absz.png']}]}
    return [panel]


@prepare_result
class BSEResult(ASRResult):

    E_B: float
    bse_alphax_w: typing.List[float]
    bse_alphay_w: typing.List[float]
    bse_alphaz_w: typing.List[float]

    key_descriptions = {
        "E_B": ('The exciton binding energy from the Bethe–Salpeter '
                'equation (BSE) [eV].'),
        'bse_alphax_w': 'BSE polarizability x-direction.',
                        'bse_alphay_w': 'BSE polarizability y-direction.',
                        'bse_alphaz_w': 'BSE polarizability z-direction.'}

    formats = {"ase_webpanel": BSEWebpanel}


######### charge_neutrality #########

# all the following are just plotting functionalities
def plot_formation_scf(row, fname):
    """Plot formation energy diagram and SC Fermi level wrt. VBM."""
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.charge_neutrality.json')
    gap = data['gap']
    comparison = fname.split('neutrality-')[-1].split('.png')[0]
    fig, ax = plt.subplots()
    for j, condition in enumerate(data['scresults']):
        if comparison == condition['condition']:
            ef = condition['efermi_sc']
            for i, defect in enumerate(condition['defect_concentrations']):
                name = defect['defect_name']
                def_type = name.split('_')[0]
                def_name = name.split('_')[-1]
                namestring = f"{def_type}$_\\{'mathrm{'}{def_name}{'}'}$"
                array = np.zeros((len(defect['concentrations']), 2))
                for num, conc_tuple in enumerate(defect['concentrations']):
                    q = conc_tuple[1]
                    eform = conc_tuple[2]
                    array[num, 0] = eform + q * (-ef)
                    array[num, 1] = q
                array = array[array[:, 1].argsort()[::-1]]
                # plot_background(ax, array)
                plot_lowest_lying(ax, array, ef, gap, name=namestring, color=f'C{i}')
            draw_band_edges(ax, gap)
            set_limits(ax, gap)
            draw_ef(ax, ef)
            set_labels_and_legend(ax, comparison)

    plt.tight_layout()
    plt.savefig(fname)


def set_labels_and_legend(ax, title):
    ax.set_xlabel(r'$E_\mathrm{F} - E_{\mathrm{VBM}}$ [eV]')
    ax.set_ylabel(f'$E^f$ [eV]')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(0.5, 1.1), ncol=5, loc='lower center')


def draw_ef(ax, ef):
    ax.axvline(ef, color='red', linestyle='dotted',
               label=r'$E_\mathrm{F}^{\mathrm{sc}}$')


def set_limits(ax, gap):
    ax.set_xlim(0 - gap / 10., gap + gap / 10.)


def get_min_el(array):
    elements = []
    for i in range(len(array)):
        elements.append(array[i, 0])
    for i, el in enumerate(elements):
        if el == min(elements):
            return i


def get_crossing_point(y1, y2, q1, q2):
    """
    Calculate the crossing point between two charge states.

    f1 = y1 + x * q1
    f2 = y2 + x * q2
    x * (q1 - q2) = y2 - y1
    x = (y2 - y1) / (q1 - q2)
    """
    return (y2 - y1) / float(q1 - q2)


def clean_array(array):
    index = get_min_el(array)

    return array[index:, :]


def get_y(x, array, index):
    q = array[index, 1]

    return q * x + array[index, 0]


def get_last_element(array, x_axis, y_axis, gap):
    y_cbms = []
    for i in range(len(array)):
        q = array[i, 1]
        eform = array[i, 0]
        y_cbms.append(q * gap + eform)

    x_axis.append(gap)
    y_axis.append(min(y_cbms))

    return x_axis, y_axis


def get_line_segment(array, index, x_axis, y_axis, gap):
    xs = []
    ys = []
    for i in range(len(array)):
        if i > index:
            y1 = array[index, 0]
            q1 = array[index, 1]
            y2 = array[i, 0]
            q2 = array[i, 1]
            crossing = get_crossing_point(y1, y2, q1, q2)
            xs.append(crossing)
            ys.append(q1 * crossing + y1)
        else:
            crossing = 1000
            xs.append(gap + 10)
            ys.append(crossing)
    min_index = index + 1
    for i, x in enumerate(xs):
        q1 = array[index, 1]
        y1 = array[index, 0]
        if x == min(xs) and x > 0 and x < gap:
            min_index = i
            x_axis.append(xs[min_index])
            y_axis.append(q1 * xs[min_index] + y1)

    return min_index, x_axis, y_axis


def plot_background(ax, array_in, gap):
    for i in range(len(array_in)):
        q = array_in[i, 1]
        eform = array_in[i, 0]
        y0 = eform
        y1 = eform + q * gap
        ax.plot([0, gap], [y0, y1], color='grey',
                alpha=0.2)


def plot_lowest_lying(ax, array_in, ef, gap, name, color):
    array_tmp = array_in.copy()
    array_tmp = clean_array(array_tmp)
    xs = [0]
    ys = [array_tmp[0, 0]]
    index, xs, ys = get_line_segment(array_tmp, 0, xs, ys, gap)
    for i in range(len(array_tmp)):
        if len(array_tmp[:, 0]) <= 1:
            break
        index, xs, ys = get_line_segment(array_tmp, index, xs, ys, gap)
        if index == len(array_tmp):
            break
    xs, ys = get_last_element(array_tmp, xs, ys, gap)
    ax.plot(xs, ys, color=color, label=name)
    ax.set_xlabel(r'$E_\mathrm{F}$ [eV]')


def draw_band_edges(ax, gap):
    ax.axvline(0, color='black')
    ax.axvline(gap, color='black')
    ax.axvspan(-100, 0, alpha=0.5, color='grey')
    ax.axvspan(gap, 100, alpha=0.5, color='grey')


def get_overview_tables(scresult, result, unitstring):
    ef = scresult.efermi_sc
    gap = result.gap
    if ef < (gap / 4.):
        dopability = '<b style="color:red;">p-type</b>'
    elif ef > (3 * gap / 4.):
        dopability = '<b style="color:blue;">n-type</b>'
    else:
        dopability = 'intrinsic'

    # get strength of p-/n-type dopability
    if ef < 0:
        ptype_val = '100+'
        ntype_val = '0'
    elif ef > gap:
        ptype_val = '0'
        ntype_val = '100+'
    else:
        ptype_val = int((1 - ef / gap) * 100)
        ntype_val = int((100 - ptype_val))
    pn_strength = f'{ptype_val:3}% / {ntype_val:3}%'
    pn = describe_entry(
        'p-type / n-type balance',
        'Balance of p-/n-type dopability in percent '
        f'(normalized wrt. band gap) at T = {int(result.temperature):d} K.'
        + dl(
            [
                [
                    '100/0',
                    code('if E<sub>F</sub> at VBM')
                ],
                [
                    '0/100',
                    code('if E<sub>F</sub> at CBM')
                ],
                [
                    '50/50',
                    code('if E<sub>F</sub> at E<sub>gap</sub> * 0.5')
                ]
            ],
        )
    )

    is_dopable = describe_entry(
        'Intrinsic doping type',
        'Is the material intrinsically n-type, p-type or intrinsic at '
        f'T = {int(result.temperature):d} K?'
        + dl(
            [
                [
                    'p-type',
                    code('if E<sub>F</sub> < 0.25 * E<sub>gap</sub>')
                ],
                [
                    'n-type',
                    code('if E<sub>F</sub> 0.75 * E<sub>gap</sub>')
                ],
                [
                    'intrinsic',
                    code('if 0.25 * E<sub>gap</sub> < E<sub>F</sub> < '
                         '0.75 * E<sub>gap</sub>')
                ],
            ],
        )
    )

    scf_fermi = describe_entry(
        'Fermi level position',
        'Self-consistent Fermi level wrt. VBM at which charge neutrality condition is '
        f'fulfilled at T = {int(result.temperature):d} K [eV].')

    scf_summary = table(result, 'Charge neutrality', [])
    scf_summary['rows'].extend([[is_dopable, dopability]])
    scf_summary['rows'].extend([[scf_fermi, f'{ef:.2f} eV']])
    scf_summary['rows'].extend([[pn, pn_strength]])

    scf_overview = table(result,
                         f'Equilibrium properties @ {int(result.temperature):d} K', [])
    scf_overview['rows'].extend([[is_dopable, dopability]])
    scf_overview['rows'].extend([[scf_fermi, f'{ef:.2f} eV']])
    scf_overview['rows'].extend([[pn, pn_strength]])
    if scresult.n0 > 1e-5:
        n0 = scresult.n0
    else:
        n0 = 0
    scf_overview['rows'].extend(
        [[describe_entry('Electron carrier concentration',
                         'Equilibrium electron carrier concentration at '
                         f'T = {int(result.temperature):d} K.'),
          f'{n0:.1e} {unitstring}']])
    if scresult.p0 > 1e-5:
        p0 = scresult.p0
    else:
        p0 = 0
    scf_overview['rows'].extend(
        [[describe_entry('Hole carrier concentration',
                         'Equilibrium hole carrier concentration at '
                         f'T = {int(result.temperature):d} K.'),
          f'{p0:.1e} {unitstring}']])

    return scf_overview, scf_summary


def get_conc_table(result, element, unitstring):
    from asr.database.browser import table, describe_entry
    from asr.defectlinks import get_defectstring_from_defectinfo

    token = element['defect_name']
    from asr.defect_symmetry import DefectInfo
    defectinfo = DefectInfo(defecttoken=token)
    defectstring = get_defectstring_from_defectinfo(
        defectinfo, charge=0)  # charge is only a dummy parameter here
    # remove the charge string from the defectstring again
    clean_defectstring = defectstring.split('(charge')[0]
    scf_table = table(result, f'Eq. concentrations of '
                              f'{clean_defectstring} [{unitstring}]', [])
    for altel in element['concentrations']:
        if altel[0] > 1e1:
            scf_table['rows'].extend(
                [[describe_entry(f'<b>Charge {altel[1]:1d}</b>',
                                 description='Equilibrium concentration '
                                             'in charge state q at T = '
                                             f'{int(result.temperature):d} K.'),
                  f'<b>{altel[0]:.1e}</b>']])
        else:
            scf_table['rows'].extend(
                [[describe_entry(f'Charge {altel[1]:1d}',
                                 description='Equilibrium concentration '
                                             'in charge state q at T = '
                                             f'{int(result.temperature):d} K.'),
                  f'{altel[0]:.1e}']])

    return scf_table


def ChargeNeutralityWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        """
    Equilibrium defect energetics evaluated by solving E<sub>F</sub> self-consistently
    until charge neutrality is achieved.
    """,
        articles=[
            href("""J. Buckeridge, Equilibrium point defect and charge carrier
     concentrations in a meterial determined through calculation of the self-consistent
     Fermi energy, Comp. Phys. Comm. 244 329 (2019)""",
                 'https://doi.org/10.1016/j.cpc.2019.06.017'),
        ],
    )

    unit = result.conc_unit
    unitstring = f"cm<sup>{unit.split('^')[-1]}</sup>"
    panels = []
    for i, scresult in enumerate(result.scresults):
        condition = scresult.condition
        tables = []
        for element in scresult.defect_concentrations:
            conc_table = get_conc_table(result, element, unitstring)
            tables.append(conc_table)
        scf_overview, scf_summary = get_overview_tables(scresult, result, unitstring)
        plotname = f'neutrality-{condition}.png'
        panel = WebPanel(
            describe_entry(f'Equilibrium energetics: all defects ({condition})',
                           panel_description),
            columns=[[fig(f'{plotname}'), scf_overview], tables],
            plot_descriptions=[{'function': plot_formation_scf,
                                'filenames': [plotname]}],
            sort=25 + i)
        panels.append(panel)

    return panels


@prepare_result
class ConcentrationResult(ASRResult):
    """Container for concentration results of a specific defect."""

    defect_name: str
    concentrations: typing.List[typing.Tuple[float, float, int]]

    key_descriptions = dict(
        defect_name='Name of the defect (see "defecttoken" of DefectInfo).',
        concentrations='List of concentration tuples containing (conc., eform @ SCEF, '
                       'chargestate).')


@prepare_result
class SelfConsistentResult(ASRResult):
    """Container for results under certain chem. pot. condition."""

    condition: str
    efermi_sc: float
    n0: float
    p0: float
    defect_concentrations: typing.List[ConcentrationResult]
    dopability: str

    key_descriptions: typing.Dict[str, str] = dict(
        condition='Chemical potential condition, e.g. A-poor. '
                  'If one is poor, all other potentials are in '
                  'rich conditions.',
        efermi_sc='Self-consistent Fermi level at which charge '
                  'neutrality condition is fulfilled [eV].',
        n0='Electron carrier concentration at SC Fermi level.',
        p0='Hole carrier concentration at SC Fermi level.',
        defect_concentrations='List of ConcentrationResult containers.',
        dopability='p-/n-type or intrinsic nature of material.')


@prepare_result
class ChargeNeutralityResult(ASRResult):
    """Container for asr.charge_neutrality results."""

    scresults: typing.List[SelfConsistentResult]
    temperature: float
    gap: float
    conc_unit: str

    key_descriptions: typing.Dict[str, str] = dict(
        scresults='List of charge neutrality results for a given '
                  'chemical potential limit.',
        temperature='Temperature [K].',
        gap='Electronic band gap [eV].',
        conc_unit='Unit of calculated concentrations.')

    formats = {"ase_webpanel": ChargeNeutralityWebpanel}


######### chc #########


class Reference:
    def __init__(self, formula, hform):
        from collections import defaultdict

        self.formula = formula
        self.hform = hform
        self.Formula = Formula(self.formula)
        self.energy = self.hform * self.natoms
        self.count = defaultdict(int)
        for k, v in self.Formula.count().items():
            self.count[k] = v
        self.symbols = list(self.Formula.count().keys())

    def __str__(self):
        """
        Make string version of object.

        Represent Reference by formula and heat of formation in a tuple.
        """
        return f'({self.formula}, {self.hform})'

    def __eq__(self, other):
        """
        Equate.

        Equate Reference-object with another
        If formulas and heat of formations
        are equal.
        """
        if type(other) != Reference:
            raise ValueError("Dont compare Reference to non-Reference")
            return False
        else:
            import numpy as np
            from asr.fere import formulas_eq
            feq = formulas_eq(self.formula, other.formula)
            heq = np.allclose(self.hform, other.hform)
            return feq and heq

    def __neq__(self, other):
        """
        Not Equal.

        Equate Reference-object with another
        if formulas and heat of formations
        are equal.
        """
        return not (self == other)

    def to_elements(self):
        return list(self.Formula.count().keys())

    def to_dict(self):
        dct = {'formula': self.formula,
               'hform': self.hform}
        return dct

    def from_dict(dct):
        formula = dct['formula']
        hform = dct['hform']
        return Reference(formula, hform)

    @property
    def natoms(self):
        return sum(self.Formula.count().values())


class Intermediate:
    def __init__(self, references, mat_reference, reactant_reference):
        self.references = references
        self.mat_ref = mat_reference
        self.reactant_ref = reactant_reference
        hform, x = self._get_hform_data()
        self.hform = hform
        self._x = x

    def to_dict(self):
        refdcts = [ref.to_dict() for ref in self.references]
        matdct = self.mat_ref.to_dict()
        reactdct = self.reactant_ref.to_dict()

        dct = {'refdcts': refdcts,
               'matdct': matdct,
               'reactdct': reactdct}
        return dct

    def from_dict(dct):
        if 'refdcts' not in dct:
            return LeanIntermediate.from_dict(dct)

        refdcts = dct['refdcts']
        matdct = dct['matdct']
        reactdct = dct['reactdct']
        refs = [Reference.from_dict(dct) for dct in refdcts]
        mat = Reference.from_dict(matdct)
        react = Reference.from_dict(reactdct)

        return Intermediate(refs, mat, react)

    @property
    def label(self):
        labels = map(lambda r: r.formula, self.references)
        x_lab = zip(self._x, labels)

        def s(x):
            return str(round(x, 2))

        label = ' + '.join([s(t[0]) + t[1] for t in x_lab])
        return label

    def to_result(self):
        thns = list(map(lambda r: (r.formula, r.hform), self.references))
        strs = [f'References: {thns}',
                f'Reactant content: {self.reactant_content}',
                f'Hform: {self.hform}']

        return strs

    def _get_hform_data(self):
        import numpy as np
        # Transform each reference into a vector
        # where entry i is count of element i
        # that is present in reference
        # Solve linear equation Ax = b
        # where A is matrix from reference vectors
        # and b is vector from mat_ref

        elements = self.mat_ref.to_elements()
        if len(elements) == 1:
            assert len(self.references) == 1, f'Els:{elements}, refs: {self.references}'
            hof = self.references[0].hform
            reac = self.reactant_ref.symbols[0]
            x = self.references[0].count[reac] / self.references[0].natoms

            return hof, [x]

        def ref2vec(_ref):
            _vec = np.zeros(len(elements))
            for i, el in enumerate(elements):
                _vec[i] = _ref.count[el]

            return _vec

        A = np.array([ref2vec(ref) for ref in self.references]).T

        b = ref2vec(self.mat_ref)

        if np.allclose(np.linalg.det(A), 0):
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            err = np.sum(np.abs(A.dot(x) - b))
            if err > 1e-4:
                for ref in self.references:
                    print(ref.formula, ref.hform)
                raise ValueError(f'Could not find solution.')
        else:
            x = np.linalg.solve(A, b)

        # hforms = np.array([ref.hform for ref in self.references])
        hforms = np.array([ref.energy for ref in self.references])

        counts = np.array([sum(ref.count.values()) for ref in self.references])
        norm = x.dot(counts)

        return np.dot(x, hforms) / norm, x

    @property
    def reactant_content(self):
        counters = zip(self._x, self.references)

        rform = self.reactant_ref.formula

        total_reactants = sum(map(lambda c: c[0] * c[1].count[rform],
                                  counters))
        total_matrefs = 1

        return total_reactants / (total_reactants + total_matrefs)


class LeanIntermediate:
    def __init__(self, mat_reference, reactant_reference,
                 reference):
        self.mat_ref = mat_reference
        self.reactant_ref = reactant_reference
        self.reference = reference
        self.hform = reference.hform
        react_symbol = reactant_reference.symbols[0]
        rc = reference.count[react_symbol] / reference.natoms
        assert not np.allclose(rc, 0.0)
        self.reactant_content = rc
        self.label = str(round(1 - rc, 2)) + reference.formula

    def to_result(self):
        thns = (self.reference.formula, self.reference.hform)
        strs = [f'Reference: {thns}',
                f'Reactant content: {self.reactant_content}',
                f'Hform: {self.hform}']

        return strs

    def to_dict(self):
        dct = {}
        dct["mat_ref"] = self.mat_ref.to_dict()
        dct["react_ref"] = self.reactant_ref.to_dict()
        dct["ref"] = self.reference.to_dict()

        return dct

    def from_dict(dct):
        mat_ref = Reference.from_dict(dct["mat_ref"])
        react_ref = Reference.from_dict(dct["react_ref"])
        ref = Reference.from_dict(dct["ref"])

        return LeanIntermediate(mat_ref, react_ref, ref)


def CHCWebpanel(result, row, key_descriptions):
    from asr.database.browser import fig as asrfig

    fname = 'convexhullcut.png'

    panel = {'title': 'Convex Hull Cut',
             'columns': [[asrfig(fname)]],
             'plot_descriptions':
             [{'function': chcut_plot,
               'filenames': [fname]}]}

    return [panel]


def filrefs(refs):
    from asr.fere import formulas_eq
    nrefs = []
    visited = []
    for (form, v) in refs:
        seen = False
        for x in visited:
            if formulas_eq(form, x):
                seen = True
                break

        if seen:
            continue
        visited.append(form)

        vals = list(filter(lambda t: formulas_eq(t[0], form), refs))

        minref = min(vals, key=lambda t: t[1])

        nrefs.append(minref)

    return nrefs


def chcut_plot(row, fname):
    import matplotlib.pyplot as plt
    from ase import Atoms

    data = row.data.get('results-asr.chc.json')
    mat_ref = Reference.from_dict(data['_matref'])

    if len(mat_ref.symbols) <= 2:
        refs = filrefs(data.get('_refs'))
        nrefs = []

        for (form, v) in refs:
            atoms = Atoms(form)
            e = v * len(atoms)
            nrefs.append((form, e))

        from ase.phasediagram import PhaseDiagram
        pd = PhaseDiagram(nrefs, verbose=False)
        plt.figure(figsize=(4, 3), dpi=150)
        pd.plot(ax=plt.gca(), dims=2, show=False)
        plt.savefig("./chcconvexhull.png")
        plt.close()

    mat_ref = Reference.from_dict(data['_matref'])
    reactant_ref = Reference.from_dict(data['_reactant_ref'])
    intermediates = [Intermediate.from_dict(im)
                     for im in data['_intermediates']]
    xs = list(map(lambda im: im.reactant_content, intermediates))
    es = list(map(lambda im: im.hform, intermediates))
    xs_es_ims = list(zip(xs, es, intermediates))
    xs_es_ims = sorted(xs_es_ims, key=lambda t: t[0])
    xs, es, ims = [list(x) for x in zip(*xs_es_ims)]
    labels = list(map(lambda im: im.label, ims))

    labels = [mat_ref.formula] + labels + [reactant_ref.formula]
    allxs = [0.0] + xs + [1.0]
    allxs = [round(x, 2) for x in allxs]
    labels = ['\n' + l if i % 2 == 1 else l for i, l in enumerate(labels)]
    labels = [f'{allxs[i]}\n' + l for i, l in enumerate(labels)]
    plt.plot([mat_ref.hform] + es + [0.0])
    plt.gca().set_xticks(range(len(labels)))
    plt.gca().set_xticklabels(labels)
    plt.xlabel(f'{reactant_ref.formula} content')
    plt.ylabel(f"Heat of formation")
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')


@prepare_result
class CHCResult(ASRResult):
    intermediates: List[Intermediate]
    material_info: str  # Reference?
    reactant: str
    mu_measure: float
    _matref: dict
    _intermediates: List[dict]
    _reactant_ref: dict
    _refs: List[Tuple[Formula, float]]

    key_descriptions = dict(
        intermediates='List of intermediates along convex hull cut.',
        material_info='Reference',
        reactant='Name of reactant to calculate cut against. E.g. "O".',
        mu_measure='Mu stability measure.',
        _matref='Material references.',
        _intermediates='List of intermediates.',
        _reactant_ref='Reference for reactant.',
        _refs='(formula, hform) list of relevant references.',
    )

    formats = {'ase_webpanel': CHCWebpanel}


######### convex_hull #########
eform_description = """\
The heat of formation (ΔH) is the internal energy of a compound relative to
the standard states of the constituent elements at T=0 K."""


ehull_description = """\
The energy above the convex hull is the internal energy relative to the most
stable (possibly mixed) phase of the constituent elements at T=0 K."""


# This is for the c2db Summary panel.  We actually define most of that panel
# in the structureinfo.py
def ehull_table_rows(row, key_descriptions):
    ehull_table = table(row, 'Stability', ['ehull', 'hform'], key_descriptions)

    # We have to magically hack a description into the arbitrarily
    # nested "table" *grumble*:
    rows = ehull_table['rows']
    if len(rows) == 2:
        # ehull and/or hform may be missing if we run tests.
        # Dangerous and hacky, as always.
        rows[0][0] = describe_entry(rows[0][0], ehull_long_description)
        rows[1][0] = describe_entry(rows[1][0], eform_description)
    return ehull_table


def convex_hull_tables(row: AtomsRow) -> List[Dict[str, Any]]:
    data = row.data['results-asr.convex_hull.json']

    references = data.get('references', [])
    tables = {}
    for reference in references:
        tables[reference['title']] = []

    for reference in sorted(references, reverse=True,
                            key=lambda x: x['hform']):
        name = reference['name']
        matlink = reference['link']
        if reference['uid'] != row.uid:
            name = f'<a href="{matlink}">{name}</a>'
        e = reference['hform']
        tables[reference['title']].append([name, '{:.2f} eV/atom'.format(e)])

    final_tables = []
    for title, rows in tables.items():
        final_tables.append({'type': 'table',
                             'header': [title, ''],
                             'rows': rows})
    return final_tables


def get_hull_energies(pd: PhaseDiagram):
    hull_energies = []
    for ref in pd.references:
        count = ref[0]
        refenergy = ref[1]
        natoms = ref[3]
        decomp_energy, indices, coefs = pd.decompose(**count)
        ehull = (refenergy - decomp_energy) / natoms
        hull_energies.append(ehull)

    return hull_energies


class ObjectHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = patches.Polygon(
            [
                [x0, y0],
                [x0, y0 + height],
                [x0 + 3 / 4 * width, y0 + height],
                [x0 + 1 / 4 * width, y0],
            ],
            closed=True, facecolor='C2',
            edgecolor='none', lw=3,
            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        patch = patches.Polygon(
            [
                [x0 + width, y0],
                [x0 + 1 / 4 * width, y0],
                [x0 + 3 / 4 * width, y0 + height],
                [x0 + width, y0 + height],
            ],
            closed=True, facecolor='C3',
            edgecolor='none', lw=3,
            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


def convex_plot(row, fname, thisrow):
    from ase.phasediagram import PhaseDiagram
    import matplotlib.pyplot as plt

    data = row.data['results-asr.convex_hull.json']

    count = row.count_atoms()
    if not (2 <= len(count) <= 3):
        return

    references = data['references']

    pdrefs = []
    legends = []
    sizes = []

    for reference in references:
        h = reference['natoms'] * reference['hform']
        pdrefs.append((reference['formula'], h))
        legend = reference.get('legend')
        if legend and legend not in legends:
            legends.append(legend)
        if legend in legends:
            idlegend = legends.index(reference['legend'])
            size = (3 * idlegend + 3)**2
        else:
            size = 2
        sizes.append(size)
    sizes = np.array(sizes)

    pd = PhaseDiagram(pdrefs, verbose=False)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.gca()

    legendhandles = []

    for it, label in enumerate(['On hull', 'off hull']):
        handle = ax.fill_between([], [],
                                 color=f'C{it + 2}', label=label)
        legendhandles.append(handle)

    for it, legend in enumerate(legends):
        handle = ax.scatter([], [], facecolor='none', marker='o',
                            edgecolor='k', label=legend, s=(3 + it * 3)**2)
        legendhandles.append(handle)

    hull_energies = get_hull_energies(pd)

    if len(count) == 2:
        xcoord, energy, _, hull, simplices, xlabel, ylabel = pd.plot2d2()
        hull = np.array(hull_energies) < 0.005
        edgecolors = np.array(['C2' if hull_energy < 0.005 else 'C3'
                               for hull_energy in hull_energies])
        for i, j in simplices:
            ax.plot(xcoord[[i, j]], energy[[i, j]], '-', color='C0')
        names = [ref['label'] for ref in references]

        if row.hform < 0:
            mask = energy < 0.005
            energy = energy[mask]
            xcoord = xcoord[mask]
            edgecolors = edgecolors[mask]
            hull = hull[mask]
            names = [name for name, m in zip(names, mask) if m]
            sizes = sizes[mask]

        xcoord0 = xcoord[~hull]
        energy0 = energy[~hull]
        ax.scatter(
            xcoord0, energy0,
            # x[~hull], e[~hull],
            facecolor='none', marker='o',
            edgecolor=np.array(edgecolors)[~hull], s=sizes[~hull],
            zorder=9)

        ax.scatter(
            xcoord[hull], energy[hull],
            facecolor='none', marker='o',
            edgecolor=np.array(edgecolors)[hull], s=sizes[hull],
            zorder=10)

        # ax.scatter(x, e, facecolor='none', marker='o', edgecolor=colors)

        delta = energy.ptp() / 30
        for a, b, name, on_hull in zip(xcoord, energy, names, hull):
            va = 'center'
            ha = 'left'
            dy = 0
            dx = 0.02
            ax.text(a + dx, b + dy, name, ha=ha, va=va)

        A, B = pd.symbols
        ax.set_xlabel('{}$_{{1-x}}${}$_x$'.format(A, B))
        ax.set_ylabel(r'$\Delta H$ [eV/atom]')

        # Circle this material
        ymin = energy.min()
        ax.axis(xmin=-0.1, xmax=1.1, ymin=ymin - 2.5 * delta)
        newlegendhandles = [(legendhandles[0], legendhandles[1]),
                            *legendhandles[2:]]

        plt.legend(
            newlegendhandles,
            [r'$E_\mathrm{h} {^</_>}\, 5 \mathrm{meV}$',
             *legends], loc='lower left', handletextpad=0.5,
            handler_map={tuple: ObjectHandler()},
        )
    else:
        x, y, _, hull, simplices = pd.plot2d3()

        hull = np.array(hull)
        hull = np.array(hull_energies) < 0.005
        names = [ref['label'] for ref in references]
        latexnames = [
            format(
                Formula(name.split(' ')[0]).reduce()[0],
                'latex'
            )
            for name in names
        ]
        for i, j, k in simplices:
            ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-', color='lightblue')
        edgecolors = ['C2' if hull_energy < 0.005 else 'C3'
                      for hull_energy in hull_energies]
        ax.scatter(
            x[~hull], y[~hull],
            facecolor='none', marker='o',
            edgecolor=np.array(edgecolors)[~hull], s=sizes[~hull],
            zorder=9,
        )

        ax.scatter(
            x[hull], y[hull],
            facecolor='none', marker='o',
            edgecolor=np.array(edgecolors)[hull], s=sizes[hull],
            zorder=10,
        )

        printed_names = set()
        thisformula = Formula(thisrow.formula)
        thisname = format(thisformula, 'latex')
        comps = thisformula.count().keys()
        for a, b, name, on_hull, hull_energy in zip(
                x, y, latexnames, hull, hull_energies):
            if name in [
                    thisname, *comps,
            ] and name not in printed_names:
                printed_names.add(name)
                ax.text(a - 0.02, b, name, ha='right', va='top')

        newlegendhandles = [(legendhandles[0], legendhandles[1]),
                            *legendhandles[2:]]
        plt.legend(
            newlegendhandles,
            [r'$E_\mathrm{h} {^</_>}\, 5 \mathrm{meV}$',
             *legends], loc='upper right', handletextpad=0.5,
            handler_map={tuple: ObjectHandler()},
        )
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def ConvexHullWebpanel(result, row, key_descriptions):
    panel_description = make_panel_description(
        f'{eform_description}\n\n{ehull_description}',
        articles=['C2DB'],
    )
    hulltable1 = table(row,
                       'Stability',
                       ['hform', 'ehull'],
                       key_descriptions)
    hulltables = convex_hull_tables(row)
    panel = {
        'title': describe_entry(
            'Thermodynamic stability', panel_description),
        'columns': [[fig('convex-hull.png')],
                    [hulltable1] + hulltables],
        'plot_descriptions': [{'function':
                               functools.partial(convex_plot, thisrow=row),
                               'filenames': ['convex-hull.png']}],
        'sort': 1,
    }

    return [panel]


# XXX This string is hardcoded also in c2db's search html file in cmr
# repository (with different formatting).
# cmr could probably import the string from here instead.
ehull_long_description = """\
The energy above the convex hull (or the decomposition energy) is the main
descriptor for thermodynamic stability. It represents the energy/atom of the
material relative to the most stable, possibly mixed phase of the material.
The latter is evaluated using a \
<a href="https://cmrdb.fysik.dtu.dk/oqmd123/">reference database of bulk \
materials</a>.
For more information see Sec. 2.3 in \
<a href="https://iopscience.iop.org/article/10.1088/2053-1583/aacfc1"> \
Haastrup <i>et al</i>.</a>
"""


@prepare_result
class ConvexHullResult(ASRResult):

    ehull: float
    hform: float
    references: List[dict]
    thermodynamic_stability_level: str
    coefs: Optional[List[float]]
    indices: Optional[List[int]]
    key_descriptions = {
        "ehull": "Energy above convex hull [eV/atom].",
        "hform": "Heat of formation [eV/atom].",
        "thermodynamic_stability_level": "Thermodynamic stability level.",
        "references": "List of relevant references.",
        "indices":
        "Indices of references that this structure will decompose into.",
        "coefs": "Fraction of decomposing references (see indices doc).",
    }

    formats = {"ase_webpanel": ConvexHullWebpanel}


######### defect symmetry #########
class Level:
    """Class to draw a single defect state level in the gap."""

    def __init__(self, energy, spin, deg, off, size=0.05, ax=None):
        self.size = size
        self.energy = energy
        self.ax = ax
        self.spin = spin
        self.deg = deg
        assert deg in [1, 2], ('only degeneracies up to two are '
                               'implemented!')
        self.off = off
        self.relpos = self.get_relative_position(self.spin, self.deg, self.off)

    def get_relative_position(self, spin, deg, off):
        """Set relative position of the level based on spin, degeneracy and offset."""
        xpos_deg = [[2 / 12, 4 / 12], [8 / 12, 10 / 12]]
        xpos_nor = [1 / 4, 3 / 4]
        if deg == 2:
            relpos = xpos_deg[spin][off]
        elif deg == 1:
            relpos = xpos_nor[spin]

        return relpos

    def draw(self):
        """Draw the defect state according to spin and degeneracy."""
        pos = [self.relpos - self.size, self.relpos + self.size]
        self.ax.plot(pos, [self.energy] * 2, '-k')

    def add_occupation(self, length):
        """Draw an arrow if the defect state if occupied."""
        updown = [1, -1][self.spin]
        self.ax.arrow(self.relpos,
                      self.energy - updown * length / 2,
                      0,
                      updown * length,
                      head_width=0.01,
                      head_length=length / 5, fc='C3', ec='C3')

    def add_label(self, label, static=None):
        """Add symmetry label of the irrep of the point group."""
        shift = self.size / 5
        labelcolor = 'C3'
        if static is None:
            labelstr = label.lower()
            splitstr = list(labelstr)
            if len(splitstr) == 2:
                labelstr = f'{splitstr[0]}$_{splitstr[1]}$'
        else:
            labelstr = 'a'

        if (self.off == 0 and self.spin == 0):
            xpos = self.relpos - self.size - shift
            ha = 'right'
        if (self.off == 0 and self.spin == 1):
            xpos = self.relpos + self.size + shift
            ha = 'left'
        if (self.off == 1 and self.spin == 0):
            xpos = self.relpos - self.size - shift
            ha = 'right'
        if (self.off == 1 and self.spin == 1):
            xpos = self.relpos + self.size + shift
            ha = 'left'
        self.ax.text(xpos,
                     self.energy,
                     labelstr,
                     va='center', ha=ha,
                     size=12,
                     color=labelcolor)


reference = """\
S. Kaappa et al. Point group symmetry analysis of the electronic structure
of bare and protected nanocrystals, J. Phys. Chem. A, 122, 43, 8576 (2018)"""

panel_description = make_panel_description(
    """
Analysis of defect states localized inside the pristine bandgap (energetics and
 symmetry).
""",
    articles=[
        href(reference, 'https://doi.org/10.1021/acs.jpca.8b07923'),
    ],
)


def get_summary_table(result, row):
    from asr.database.browser import table
    from asr.structureinfo import get_spg_href, describe_pointgroup_entry

    spglib = get_spg_href('https://spglib.github.io/spglib/')
    basictable = table(row, 'Defect properties', [])
    pg_string = result.defect_pointgroup
    pg_strlist = list(pg_string)
    sub = ''.join(pg_strlist[1:])
    pg_string = f'{pg_strlist[0]}<sub>{sub}</sub>'
    pointgroup = describe_pointgroup_entry(spglib)
    basictable['rows'].extend(
        [[pointgroup, pg_string]])

    return basictable


def get_number_of_rows(res, spin, vbm, cbm):
    counter = 0
    for i in range(len(res)):
        if (int(res[i]['spin']) == spin
           and res[i]['energy'] < cbm
           and res[i]['energy'] > vbm):
            counter += 1

    return counter


def get_matrixtable_array(state_results, vbm, cbm, ef,
                          spin, style):
    Nrows = get_number_of_rows(state_results, spin, vbm, cbm)
    state_array = np.empty((Nrows, 5), dtype='object')
    rowlabels = []
    spins = []
    energies = []
    symlabels = []
    accuracies = []
    loc_ratios = []
    for i, row in enumerate(state_results):
        rowname = f"{int(state_results[i]['state']):.0f}"
        label = str(state_results[i]['best'])
        labelstr = label.lower()
        splitstr = list(labelstr)
        if len(splitstr) == 2:
            labelstr = f'{splitstr[0]}<sub>{splitstr[1]}</sub>'
        if state_results[i]['energy'] < cbm and state_results[i]['energy'] > vbm:
            if int(state_results[i]['spin']) == spin:
                rowlabels.append(rowname)
                spins.append(f"{int(state_results[i]['spin']):.0f}")
                energies.append(f"{state_results[i]['energy']:.2f}")
                if style == 'symmetry':
                    symlabels.append(labelstr)
                    accuracies.append(f"{state_results[i]['error']:.2f}")
                    loc_ratios.append(f"{state_results[i]['loc_ratio']:.2f}")
    state_array = np.empty((Nrows, 5), dtype='object')
    rowlabels.sort(reverse=True)

    for i in range(Nrows):
        state_array[i, 1] = spins[i]
        if style == 'symmetry':
            state_array[i, 0] = symlabels[i]
            state_array[i, 2] = accuracies[i]
            state_array[i, 3] = loc_ratios[i]
        state_array[i, 4] = energies[i]
    state_array = state_array[state_array[:, -1].argsort()]

    return state_array, rowlabels


def get_symmetry_tables(state_results, vbm, cbm, row, style):
    state_tables = []
    gsdata = row.data.get('results-asr.gs.json')
    eref = row.data.get('results-asr.get_wfs.json')['eref']
    ef = gsdata['efermi'] - eref

    E_hls = []
    for spin in range(2):
        state_array, rowlabels = get_matrixtable_array(
            state_results, vbm, cbm, ef, spin, style)
        if style == 'symmetry':
            delete = [2]
            columnlabels = ['Symmetry',
                            # 'Spin',
                            'Localization ratio',
                            'Energy']
        elif style == 'state':
            delete = [0, 2, 3]
            columnlabels = [  # 'Spin',
                'Energy']

        N_homo = 0
        N_lumo = 0
        for i in range(len(state_array)):
            if float(state_array[i, 4]) > ef:
                N_lumo += 1

        E_homo = vbm
        E_lumo = cbm
        for i in range(len(state_array)):
            if float(state_array[i, 4]) > ef:
                rowlabels[i] = f'LUMO + {N_lumo - 1}'
                N_lumo = N_lumo - 1
                if N_lumo == 0:
                    rowlabels[i] = 'LUMO'
                    E_lumo = float(state_array[i, 4])
            elif float(state_array[i, 4]) <= ef:
                rowlabels[i] = f'HOMO — {N_homo}'
                if N_homo == 0:
                    rowlabels[i] = 'HOMO'
                    E_homo = float(state_array[i, 4])
                N_homo = N_homo + 1
        E_hl = E_lumo - E_homo
        E_hls.append(E_hl)

        state_array = np.delete(state_array, delete, 1)
        headerlabels = [f'Orbitals in spin channel {spin}',
                        *columnlabels]

        rows = []
        state_table = {'type': 'table',
                       'header': headerlabels}
        for i in range(len(state_array)):
            if style == 'symmetry':
                rows.append((rowlabels[i],
                             # state_array[i, 0],
                             state_array[i, 1],
                             describe_entry(state_array[i, 2],
                                            'The localization ratio is defined as the '
                                            'volume of the cell divided by the integral'
                                            ' of the fourth power of the '
                                            'wavefunction.'),
                             f'{state_array[i, 3]} eV'))
            elif style == 'state':
                rows.append((rowlabels[i],
                             # state_array[i, 0],
                             f'{state_array[i, 1]} eV'))

        state_table['rows'] = rows
        state_tables.append(state_table)

    transition_table = get_transition_table(row, E_hls)

    return state_tables, transition_table


def get_transition_table(row, E_hls):
    """Create table for HOMO-LUMO transition in both spin channels."""
    from asr.database.browser import table

    transition_table = table(row, 'Kohn—Sham HOMO—LUMO gap', [])
    for i, element in enumerate(E_hls):
        transition_table['rows'].extend(
            [[describe_entry(f'Spin {i}',
                             f'KS HOMO—LUMO gap for spin {i} channel.'),
              f'{element:.2f} eV']])

    return transition_table


def get_spin_data(data, spin):
    """Create symmetry result only containing entries for one spin channel."""
    spin_data = []
    for sym in data.data['symmetries']:
        if int(sym.spin) == spin:
            spin_data.append(sym)

    return spin_data


def draw_levels_occupations_labels(ax, spin, spin_data, ecbm, evbm, ef,
                                   gap, levelflag):
    """Loop over all states in the gap and plot the levels.

    This function loops over all states in the gap of a given spin
    channel, and dravs the states with labels. If there are
    degenerate states, it makes use of the degeneracy_counter, i.e. if two
    degenerate states follow after each other, one of them will be drawn
    on the left side (degoffset=0, degeneracy_counter=0), the degeneracy
    counter will be increased by one and the next degenerate state will be
    drawn on the right side (degoffset=1, degeneracy_counter=1). Since we
    only deal with doubly degenerate states here, the degeneracy counter
    will be set to zero again after drawing the second degenerate state.

    For non degenerate states, i.e. deg = 1, all states will be drawn
    in the middle and the counter logic is not needed.
    """
    # initialize degeneracy counter and offset
    degeneracy_counter = 0
    degoffset = 0
    for sym in spin_data:
        energy = sym.energy
        is_inside_gap = evbm < energy < ecbm
        if is_inside_gap:
            spin = int(sym.spin)
            irrep = sym.best
            # only do drawing left and right if levelflag, i.e.
            # if there is a symmetry analysis to evaluate degeneracies
            if levelflag:
                deg = [1, 2]['E' in irrep]
            else:
                deg = 1
                degoffset = 1
            # draw draw state on the left hand side
            if deg == 2 and degeneracy_counter == 0:
                degoffset = 0
                degeneracy_counter = 1
            # draw state on the right hand side, set counter to zero again
            elif deg == 2 and degeneracy_counter == 1:
                degoffset = 1
                degeneracy_counter = 0
            # intitialize and draw the energy level
            lev = Level(energy, ax=ax, spin=spin, deg=deg,
                        off=degoffset)
            lev.draw()
            # add occupation arrow if level is below E_F
            if energy <= ef:
                lev.add_occupation(length=gap / 15.)
            # draw label based on irrep
            if levelflag:
                static = None
            else:
                static = 'A'
            lev.add_label(irrep, static=static)


def draw_band_edge(energy, edge, color, *, offset=2, ax):
    if edge == 'vbm':
        eoffset = energy - offset
        elabel = energy - offset / 2
    elif edge == 'cbm':
        eoffset = energy + offset
        elabel = energy + offset / 2

    ax.plot([0, 1], [energy] * 2, color='black', zorder=1)
    ax.fill_between([0, 1], [energy] * 2, [eoffset] * 2, color='grey', alpha=0.5)
    ax.text(0.5, elabel, edge.upper(), color='w', weight='bold', ha='center',
            va='center', fontsize=12)


def plot_gapstates(row, fname):
    from matplotlib import pyplot as plt

    data = row.data.get('results-asr.defect_symmetry.json')
    gsdata = row.data.get('results-asr.gs.json')

    fig, ax = plt.subplots()

    # extract pristine data
    evbm = data.pristine.vbm
    ecbm = data.pristine.cbm
    gap = data.pristine.gap
    eref = row.data.get('results-asr.get_wfs.json')['eref']
    ef = gsdata['efermi'] - eref

    # Draw band edges
    draw_band_edge(evbm, 'vbm', 'C0', offset=gap / 5, ax=ax)
    draw_band_edge(ecbm, 'cbm', 'C1', offset=gap / 5, ax=ax)

    levelflag = data.symmetries[0].best is not None
    # draw the levels with occupations, and labels for both spins
    for spin in [0, 1]:
        spin_data = get_spin_data(data, spin)
        draw_levels_occupations_labels(ax, spin, spin_data, ecbm, evbm,
                                       ef, gap, levelflag)

    ax1 = ax.twinx()
    ax.set_xlim(0, 1)
    ax.set_ylim(evbm - gap / 5, ecbm + gap / 5)
    ax1.set_ylim(evbm - gap / 5, ecbm + gap / 5)
    ax1.plot([0, 1], [ef] * 2, '--k')
    ax1.set_yticks([ef])
    ax1.set_yticklabels([r'$E_\mathrm{F}$'])
    ax.set_xticks([])
    ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def DefectSymmetryWebpanel(result, row, key_descriptions):
    from asr.database.browser import (WebPanel,
                                      describe_entry,
                                      fig)

    description = describe_entry('One-electron states', panel_description)
    basictable = get_summary_table(result, row)

    vbm = result.pristine['vbm']
    cbm = result.pristine['cbm']
    if result.symmetries[0]['best'] is None:
        warnings.warn("no symmetry analysis present for this defect. "
                      "Only plot gapstates!", UserWarning)
        style = 'state'
    else:
        style = 'symmetry'

    state_tables, transition_table = get_symmetry_tables(
        result.symmetries, vbm, cbm, row, style=style)
    panel = WebPanel(description,
                     columns=[[state_tables[0],
                               fig('ks_gap.png')],
                              [state_tables[1], transition_table]],
                     plot_descriptions=[{'function': plot_gapstates,
                                         'filenames': ['ks_gap.png']}],
                     sort=30)

    summary = {'title': 'Summary',
               'columns': [[basictable, transition_table], []],
               'sort': 2}

    return [panel, summary]


@prepare_result
class IrrepResult(ASRResult):
    """Container for results of an individual irreproducible representation."""

    sym_name: str
    sym_score: float

    key_descriptions: typing.Dict[str, str] = dict(
        sym_name='Name of the irreproducible representation.',
        sym_score='Score of the respective representation.')


@prepare_result
class SymmetryResult(ASRResult):
    """Container for symmetry results for a given state."""

    irreps: typing.List[IrrepResult]
    best: str
    error: float
    loc_ratio: float
    state: int
    spin: int
    energy: float

    key_descriptions: typing.Dict[str, str] = dict(
        irreps='List of irreproducible representations and respective scores.',
        best='Irreproducible representation with the best score.',
        error='Error of identification of the best irreproducible representation.',
        loc_ratio='Localization ratio for a given state.',
        state='Index of the analyzed state.',
        spin='Spin of the analyzed state (0 or 1).',
        energy='Energy of specific state aligned to pristine semi-core state [eV].'
    )


@prepare_result
class PristineResult(ASRResult):
    """Container for pristine band edge results."""

    vbm: float
    cbm: float
    gap: float

    key_descriptions: typing.Dict[str, str] = dict(
        vbm='Energy of the VBM (ref. to the vacuum level in 2D) [eV].',
        cbm='Energy of the CBM (ref. to the vacuum level in 2D) [eV].',
        gap='Energy of the bandgap [eV].')


@prepare_result
class DefectSymmetryResult(ASRResult):
    """Container for main results for asr.analyze_state."""

    defect_pointgroup: str
    defect_center: typing.Tuple[float, float, float]
    defect_name: str
    symmetries: typing.List[SymmetryResult]
    pristine: PristineResult

    key_descriptions: typing.Dict[str, str] = dict(
        defect_pointgroup='Point group in Schoenflies notation.',
        defect_center='Position of the defect [Å, Å, Å].',
        defect_name='Name of the defect ({type}_{position})',
        symmetries='List of SymmetryResult objects for all states.',
        pristine='PristineResult container.'
    )

    formats = {'ase_webpanel': DefectSymmetryWebpanel}


#########  #########

#########  #########


######### HSE #########
from ase.spectrum.band_structure import BandStructure
def plot_bs(row,
            filename,
            *,
            bs_label,
            efermi,
            data,
            vbm,
            cbm):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    figsize = (5.5, 5)
    fontsize = 10

    path = data['bandstructure']['path']

    reference = row.get('evac')
    if reference is None:
        reference = efermi
        label = r'$E - E_\mathrm{F}$ [eV]'
    else:
        label = r'$E - E_\mathrm{vac}$ [eV]'

    emin_offset = efermi if vbm is None else vbm
    emax_offset = efermi if cbm is None else cbm
    emin = emin_offset - 3 - reference
    emax = emax_offset + 3 - reference

    e_mk = data['bandstructure']['e_int_mk'] - reference
    x, X, labels = path.get_linear_kpoint_axis()

    # with soc
    style = dict(
        color='C1',
        ls='-',
        lw=1.0,
        zorder=0)
    ax = plt.figure(figsize=figsize).add_subplot(111)
    for e_m in e_mk:
        ax.plot(x, e_m, **style)
    ax.set_ylim([emin, emax])
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylabel(label)
    ax.set_xticks(X)
    ax.set_xticklabels([lab.replace('G', r'$\Gamma$') for lab in labels])

    xlim = ax.get_xlim()
    x0 = xlim[1] * 0.01
    ax.axhline(efermi - reference, c='C1', ls=':')
    text = ax.annotate(
        r'$E_\mathrm{F}$',
        xy=(x0, efermi - reference),
        ha='left',
        va='bottom',
        fontsize=fontsize * 1.3)
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    # add KS band structure with soc
    if 'results-asr.bandstructure.json' in row.data:
        ax = add_bs_ks(row, ax, reference=row.get('evac', row.get('efermi')),
                       color=[0.8, 0.8, 0.8])

    for Xi in X:
        ax.axvline(Xi, ls='-', c='0.5', zorder=-20)

    ax.plot([], [], **style, label=bs_label)
    legend_on_top(ax, ncol=2)
    plt.savefig(filename, bbox_inches='tight')

from asr.utils.gw_hse import GWHSEInfo
class HSEInfo(GWHSEInfo):
    method_name = 'HSE06'
    name = 'hse'
    bs_filename = 'hse-bs.png'

    panel_description = make_panel_description(
        """\
The single-particle band structure calculated with the HSE06
xc-functional. The calculations are performed non-self-consistently with the
wave functions from a GGA calculation. Spin–orbit interactions are included
in post-process.""",
        articles=['C2DB'],
    )

    band_gap_adjectives = 'electronic single-particle'
    summary_sort = 11

    @staticmethod
    def plot_bs(row, filename):
        data = row.data['results-asr.hse.json']
        return plot_bs(row, filename=filename, bs_label='HSE06',
                       data=data,
                       efermi=data['efermi_hse_soc'],
                       vbm=row.get('vbm_hse'),
                       cbm=row.get('cbm_hse'))


def HSEWebpanel(result, row, key_descriptions):
    from asr.utils.gw_hse import gw_hse_webpanel
    return gw_hse_webpanel(result, row, key_descriptions, HSEInfo(row),
                           sort=12.5)


@prepare_result
class HSEResult(ASRResult):
    vbm_hse_nosoc: float
    cbm_hse_nosoc: float
    gap_dir_hse_nosoc: float
    gap_hse_nosoc: float
    kvbm_nosoc: typing.List[float]
    kcbm_nosoc: typing.List[float]
    vbm_hse: float
    cbm_hse: float
    gap_dir_hse: float
    gap_hse: float
    kvbm: typing.List[float]
    kcbm: typing.List[float]
    efermi_hse_nosoc: float
    efermi_hse_soc: float
    bandstructure: BandStructure

    key_descriptions = {
        "vbm_hse_nosoc": "Valence band maximum w/o soc. (HSE06) [eV]",
        "cbm_hse_nosoc": "Conduction band minimum w/o soc. (HSE06) [eV]",
        "gap_dir_hse_nosoc": "Direct gap w/o soc. (HSE06) [eV]",
        "gap_hse_nosoc": "Band gap w/o soc. (HSE06) [eV]",
        "kvbm_nosoc": "k-point of HSE06 valence band maximum w/o soc",
        "kcbm_nosoc": "k-point of HSE06 conduction band minimum w/o soc",
        "vbm_hse": "KVP: Valence band maximum (HSE06) [eV]",
        "cbm_hse": "KVP: Conduction band minimum (HSE06) [eV]",
        "gap_dir_hse": "KVP: Direct band gap (HSE06) [eV]",
        "gap_hse": "KVP: Band gap (HSE06) [eV]",
        "kvbm": "k-point of HSE06 valence band maximum",
        "kcbm": "k-point of HSE06 conduction band minimum",
        "efermi_hse_nosoc": "Fermi level w/o soc. (HSE06) [eV]",
        "efermi_hse_soc": "Fermi level (HSE06) [eV]",
        "bandstructure": "HSE06 bandstructure."
    }
    formats = {"ase_webpanel": HSEWebpanel}


