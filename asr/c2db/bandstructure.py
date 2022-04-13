"""Electronic band structures."""
import pathlib
import copy
from typing import Union
import asr
from asr.calculators import Calculation
from asr.core import (
    command, option, ASRResult, singleprec_dict, prepare_result,
    Selector,
)

import numpy as np
from ase.dft.kpoints import labels_from_kpts
from asr.database.browser import fig, make_panel_description, describe_entry
from asr.utils.symmetry import c2db_symmetry_eps


panel_description = make_panel_description(
    """The band structure with spin–orbit interactions is shown with the
expectation value of S_i (where i=z for non-magnetic materials and otherwise is
the magnetic easy axis) indicated by the color code.""",
    articles=['C2DB'],
)


@prepare_result
class BandstructureCalculationResult(ASRResult):

    calculation: Calculation

    key_descriptions = dict(calculation='Calculation object')


sel = Selector()
sel.name = sel.EQ('asr.c2db.bandstructure:calculate')
sel.version = sel.EQ(-1)
sel.parameters = sel.AND(
    sel.CONTAINS('emptybands'),
    sel.NOT(sel.CONTAINS('bsrestart')),
)


@asr.mutation(selector=sel)
def remove_emptybands_and_make_bsrestart(record):
    """Remove param="emptybands" and make param='bsrestart'."""
    record.parameters.bsrestart = {
        'nbands': -record.parameters.emptybands,
        'txt': 'bs.txt',
        'fixdensity': True,
        'convergence': {
            'bands': -record.parameters.emptybands // 2},
        'symmetry': 'off'
    }
    del record.parameters.emptybands
    return record


bsrestart_defaults = {
    'nbands': -20,
    'txt': 'bs.txt',
    'fixdensity': True,
    'convergence': {
        'bands': -10},
    'symmetry': 'off'
}


default_npoints = 400


@command(
    'asr.c2db.bandstructure',
)
@asr.calcopt(
    aliases=['-b', '--bsrestart'],
    help='Bandstructure Calculator params.',
    matcher=asr.matchers.EQUAL,
)
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints',
        type=int,
        help='Number of points along k-point path.')
def calculate(
        gsresult,
        bsrestart=None,
        kptpath: Union[str, None] = None,
        npoints: int = default_npoints,
) -> BandstructureCalculationResult:
    """Calculate electronic band structure."""
    calculation = gsresult.calculation

    # XXX somewhat hacky way to get the atoms.
    # We could prboably rely on them having been the input to the GS
    # calculation, and that should be more easily expressible.
    atoms = calculation.load().get_atoms()
    path = atoms.cell.bandpath(path=kptpath, npoints=npoints,
                               pbc=atoms.pbc, eps=c2db_symmetry_eps)

    bsrestart = {**bsrestart_defaults, **(bsrestart or {})}

    bsrestart['kpts'] = path
    calc = calculation.load(**bsrestart)
    atoms = calc.get_atoms()
    calc.get_potential_energy()
    calculation = calc.save(id='bs')
    return BandstructureCalculationResult.fromdata(calculation=calculation)


bs_png = 'bs.png'
bs_html = 'bs.html'


def plot_bs_html(context,
                 filename=bs_html,
                 figsize=(6.4, 6.4),
                 s=2):
    import plotly
    import plotly.graph_objs as go

    traces = []
    d = context.result
    xcname = context.xcname

    path = d['bs_nosoc']['path']
    kpts = path.kpts
    ef = d['bs_nosoc']['efermi']

    ref = context.energy_reference()
    label = ref.html_plotlabel()

    emin, emax = context.bs_energy_window()

    e_skn = d['bs_nosoc']['energies']
    shape = e_skn.shape
    cell = context.atoms.cell
    xcoords, label_xcoords, orig_labels = labels_from_kpts(kpts, cell)
    xcoords = np.vstack([xcoords] * shape[0] * shape[2])
    # colors_s = plt.get_cmap('viridis')([0, 1])  # color for sz = 0
    e_kn = np.hstack([e_skn[x] for x in range(shape[0])])
    trace = go.Scattergl(
        x=xcoords.ravel(),
        y=e_kn.T.ravel() - ref.value,
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

    xcoords, label_xcoords, orig_labels = labels_from_kpts(kpts, cell)

    shape = e_mk.shape
    perm = (-sz_mk).argsort(axis=None)
    e_mk = e_mk.ravel()[perm].reshape(shape)
    sz_mk = sz_mk.ravel()[perm].reshape(shape)
    xcoords = np.vstack([xcoords] * shape[0])
    xcoords = xcoords.ravel()[perm].reshape(shape)

    # Unicode for <S_z>
    sdir = context.spin_axis
    cbtitle = '&#x3008; <i><b>S</b></i><sub>{}</sub> &#x3009;'.format(sdir)
    trace = go.Scattergl(
        x=xcoords.ravel(),
        y=e_mk.ravel() - ref.value,
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
        y=[ef - ref.value, ef - ref.value],
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
        range=[emin - ref.value, emax - ref.value],
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


def add_bs_ks(context, ax, reference=0, color='C1'):
    """Plot with soc on ax."""
    bsrecord = context.bandstructure()
    d = bsrecord.result
    path = d['bs_soc']['path']
    e_mk = d['bs_soc']['energies']
    xcoords, label_xcoords, labels = labels_from_kpts(path.kpts,
                                                      context.atoms.cell)
    for e_k in e_mk[:-1]:
        ax.plot(xcoords, e_k - reference, color=color, zorder=-2)
    ax.lines[-1].set_label(context.xcname)
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


def plot_bs_png(context,
                filename=bs_png,
                figsize=(5.5, 5),
                s=0.5):

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.patheffects as path_effects
    from ase.spectrum.band_structure import BandStructure, BandStructurePlot

    d = context.result
    xcname = context.xcname
    eref = context.energy_reference()

    path = d['bs_nosoc']['path']
    ef_soc = d['bs_soc']['efermi']

    ref_soc = eref.value
    if context.ndim != 2:
        # XXXX this check should be ndim == 3, but we need to update GS
        # so it sets the vacuum level for ndim != 2.
        ref_nosoc = d['bs_nosoc']['efermi']
    else:
        assert eref.key == 'evac'
        ref_nosoc = ref_soc

    label = eref.mpl_plotlabel()

    e_skn = d['bs_nosoc']['energies']
    nspins = e_skn.shape[0]
    e_kn = np.hstack([e_skn[x] for x in range(nspins)])[np.newaxis]

    emin, emax = context.bs_energy_window()

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

    # XXX We do not depend on structureinfo so we cannot
    # use has_inversion_symmetry!
    colorbar = context.is_magnetic
    # colorbar = not (row.magstate == 'NM'
    #                 and getattr(row, 'has_inversion_symmetry', False))
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
        clabel=r'$\langle S_{} \rangle $'.format(context.spin_axis),
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


def webpanel(result, context):
    xcname = context.xcname

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
class Result(ASRResult):

    version: int = 0

    bs_soc: dict
    bs_nosoc: dict

    key_descriptions = \
        {
            'bs_soc': 'Bandstructure data with spin–orbit coupling.',
            'bs_nosoc': 'Bandstructure data without spin–orbit coupling.'
        }

    formats = {"webpanel2": webpanel}


sel = Selector()
sel.name = sel.EQ('asr.c2db.bandstructure:main')
sel.version = sel.EQ(-1)
sel.parameters = sel.NOT(sel.CONTAINS('bsrestart'))


@asr.mutation(selector=sel)
def set_bsrestart_from_dependencies(record):
    """Construct "bsrestart" parameters from "emptybands" parameter."""
    emptybands = (
        record.parameters.dependency_parameters[
            'asr.c2db.bandstructure:calculate']['emptybands']
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
        'asr.c2db.bandstructure:calculate']['emptybands']
    return record


@command('asr.c2db.bandstructure')
# @option('-a', '--atoms', help='Atomic structure.',
#        type=AtomsFile(), default='structure.json')
# @asr.calcopt
# @asr.calcopt(
#    aliases=['-b', '--bsrestart'],
#    help='Bandstructure Calculator params.',
#    matcher=asr.matchers.EQUAL,
# )
# @option('--kptpath', type=str, help='Custom kpoint path.')
# @option('--npoints',
#        type=int,
#        help='Number of points along k-point path.')
def postprocess(bsresult, gsresult, mag_ani, gspostprocess) -> Result:
    from ase.spectrum.band_structure import get_band_structure
    from ase.dft.kpoints import BandPath
    from asr.utils.gpw2eigs import gpw2eigs

    # bsresult = calculate(
    #    atoms=atoms,
    #    calculator=calculator,
    #    bsrestart=bsrestart,
    #    npoints=npoints,
    #    kptpath=kptpath,
    # )
    # gsresult = calculategs(atoms=atoms, calculator=calculator)
    ref = gsresult.calculation.load().get_fermi_level()
    calc = bsresult.calculation.load()
    atoms = calc.atoms
    path = calc.parameters.kpts
    if not isinstance(path, BandPath):
        if 'kpts' in path:
            # In this case path comes from a bandpath object
            path = BandPath(kpts=path['kpts'], cell=path['cell'],
                            special_points=path['special_points'],
                            path=path['labelseq'])
        else:
            path = calc.atoms.cell.bandpath(pbc=atoms.pbc,
                                            path=path['path'],
                                            npoints=path['npoints'],
                                            eps=c2db_symmetry_eps)
    bs = get_band_structure(calc=calc, path=path, reference=ref)

    results = {}
    bsresults = bs.todict()

    # Save Fermi levels
    # XXXX Why do we have "gsresult" for the gpw file and also
    # gsresults being the main/postprocess result?
    #
    # We just got the Fermi level a few lines ago!
    #
    gsresults = gspostprocess
    # gsresults = maings(atoms=atoms, calculator=calculator)
    efermi_nosoc = gsresults['gaps_nosoc']['efermi']
    bsresults['efermi'] = efermi_nosoc

    # We copy the bsresults dict because next we will add SOC
    results['bs_nosoc'] = copy.deepcopy(bsresults)  # BS with no SOC

    # Add spin orbit correction
    bsresults = bs.todict()

    # mag_ani = mag_ani_main(atoms=atoms, calculator=calculator)
    theta, phi = mag_ani.spin_angles()

    # We use a larger symmetry tolerance because we want to correctly
    # color spins which doesn't always happen due to slightly broken
    # symmetries, hence tolerance=1e-2.
    # XXX This is only compatible with GPAW
    bsfile = bsresult.calculation.paths[0]
    e_km, _, s_kvm = gpw2eigs(
        pathlib.Path(bsfile), soc=True, return_spin=True, theta=theta, phi=phi,
        symmetry_tolerance=1e-2)
    bsresults['energies'] = e_km.T
    efermi = gsresults['efermi']
    bsresults['efermi'] = efermi

    # Get spin projections for coloring of bandstructure
    path = bsresults['path']
    npoints = len(path.kpts)
    s_mvk = np.array(s_kvm.transpose(2, 1, 0))

    if s_mvk.ndim == 3:
        sz_mk = s_mvk[
            :,
            mag_ani.spin_index(),
            :]  # take x, y or z component
    else:
        sz_mk = s_mvk

    assert sz_mk.shape[1] == npoints, f'sz_mk has wrong dims, {npoints}'

    bsresults['sz_mk'] = sz_mk

    return Result.fromdata(
        bs_soc=singleprec_dict(bsresults),
        bs_nosoc=singleprec_dict(results['bs_nosoc'])
    )


class BSWorkflow:
    def __init__(self, rn, gsworkflow, *,
                 bsrestart=None, kptpath=None, npoints=default_npoints):
        self.gsworkflow = gsworkflow
        self.bs = rn.task(
            'asr.c2db.bandstructure.calculate',
            gsresult=gsworkflow.scf.output, bsrestart=bsrestart,
            kptpath=kptpath, npoints=npoints)
        self.postprocess = rn.task(
            'asr.c2db.bandstructure.postprocess',
            bsresult=self.bs.output,
            gsresult=gsworkflow.scf.output,
            mag_ani=gsworkflow.magnetic_anisotropy.output,
            gspostprocess=gsworkflow.postprocess.output)


# Temporary class for porting to htw
class BS:
    def __init__(self, gs, **kwargs):
        self.calculateresult = calculate(gsresult=gs.gsresult, **kwargs)
        self.post = postprocess(
            bsresult=self.calculateresult, gsresult=gs.gsresult,
            mag_ani=gs.mag_ani, gspostprocess=gs.post)
