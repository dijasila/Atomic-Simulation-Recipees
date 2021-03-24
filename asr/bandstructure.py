"""Electronic band structures."""
from typing import Union
from ase import Atoms
from asr.calculators import Calculation
from asr.core import (
    command, option, ASRResult, singleprec_dict, prepare_result,
    DictStr, AtomsFile, Selector, make_migration_generator,
)
from asr.gs import calculate as calculategs
from asr.gs import main as maings

from asr.database.browser import fig, make_panel_description, describe_entry

panel_description = make_panel_description(
    """The band structure with spin-orbit interactions is shown with the
expectation value of S_i (where i=z for non-magnetic materials and otherwise is
the magnetic easy axis) indicated by the color code.""",
    articles=['C2DB'],
)


@prepare_result
class BandstructureCalculationResult(ASRResult):

    calculation: Calculation

    key_descriptions = dict(calculation='Calculation object')


def remove_emptybands_and_make_bs_calculator(record):
    record.parameters.bscalculator = {
        'basis': 'dzp',
        'nbands': -record.parameters.emptybands,
        'txt': 'bs.txt',
        'fixdensity': True,
        'convergence': {
            'bands': -record.parameters.emptybands // 2},
        'symmetry': 'off'
    }
    del record.parameters.emptybands
    return record


sel = Selector()
sel.name = sel.EQ('asr.bandstructure:calculate')
sel.version = sel.EQ(-1)
sel.parameters = sel.AND(
    sel.CONTAINS('emptybands'),
    sel.NOT(sel.CONTAINS('bscalculator')),
)

make_migrations = make_migration_generator(
    selector=sel,
    function=remove_emptybands_and_make_bs_calculator,
    description='Remove param="emptybands" and make param="bscalculator"',
    uid='a0887cebb5224e1097010f1545ce766f',
)


@command(
    'asr.bandstructure',
    migrations=[make_migrations],
)
@option('-a', '--atoms', help='Atomic structure.',
        type=AtomsFile(), default='structure.json')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
@option('-b', '--bscalculator',
        help='Bandstructure Calculator params.',
        type=DictStr())
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints',
        type=int,
        help='Number of points along k-point path.')
def calculate(
        atoms: Atoms,
        calculator: dict = calculategs.defaults.calculator,
        bscalculator: dict = {
            'basis': 'dzp',
            'nbands': -20,
            'txt': 'bs.txt',
            'fixdensity': True,
            'convergence': {
                'bands': -10},
            'symmetry': 'off'
        },
        kptpath: Union[str, None] = None,
        npoints: int = 400,
) -> BandstructureCalculationResult:
    """Calculate electronic band structure."""
    if kptpath is None:
        path = atoms.cell.bandpath(npoints=npoints, pbc=atoms.pbc)
    else:
        path = atoms.cell.bandpath(path=kptpath, npoints=npoints,
                                   pbc=atoms.pbc)

    record = calculategs(atoms=atoms, calculator=calculator)
    calculation = record.result.calculation
    bscalculator['kpts'] = path
    calc = calculation.load(**bscalculator)
    calc.get_potential_energy()
    calculation = calc.save(id='bs')
    return BandstructureCalculationResult.fromdata(calculation=calculation)


def bs_pbe_html(row,
                filename='pbe-bs.html',
                figsize=(6.4, 6.4),
                show_legend=True,
                s=2):
    import plotly
    import plotly.graph_objs as go
    import numpy as np

    traces = []
    d = row.data.get('results-asr.bandstructure.json')

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
    from ase.dft.kpoints import labels_from_kpts
    xcoords, label_xcoords, orig_labels = labels_from_kpts(kpts, row.cell)
    xcoords = np.vstack([xcoords] * shape[0] * shape[2])
    # colors_s = plt.get_cmap('viridis')([0, 1])  # color for sz = 0
    e_kn = np.hstack([e_skn[x] for x in range(shape[0])])
    trace = go.Scattergl(
        x=xcoords.ravel(),
        y=e_kn.T.ravel() - reference,
        mode='markers',
        name='PBE no SOC',
        showlegend=True,
        marker=dict(size=4, color='#999999'))
    traces.append(trace)

    e_mk = d['bs_soc']['energies']
    path = d['bs_soc']['path']
    kpts = path.kpts
    ef = d['bs_soc']['efermi']
    sz_mk = d['bs_soc']['sz_mk']

    from ase.dft.kpoints import labels_from_kpts
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


def add_bs_pbe(row, ax, reference=0, color='C1'):
    """Plot pbe with soc on ax."""
    from ase.dft.kpoints import labels_from_kpts
    d = row.data.get('results-asr.bandstructure.json')
    path = d['bs_soc']['path']
    e_mk = d['bs_soc']['energies']
    xcoords, label_xcoords, labels = labels_from_kpts(path.kpts, row.cell)
    for e_k in e_mk[:-1]:
        ax.plot(xcoords, e_k - reference, color=color, zorder=-2)
    ax.lines[-1].set_label('PBE')
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


def bs_pbe(row,
           filename='pbe-bs.png',
           figsize=(5.5, 5),
           show_legend=True,
           s=0.5):

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.patheffects as path_effects
    import numpy as np
    from ase.spectrum.band_structure import BandStructure, BandStructurePlot
    d = row.data.get('results-asr.bandstructure.json')

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
    # pbe without soc
    nosoc_style = dict(
        colors=['0.8'] * e_skn.shape[0],
        label='PBE no SOC',
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
    # pbe with soc
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
    ax.plot([], [], label='PBE', color=csz0)
    plt.legend(loc='upper right')
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
    if not show_legend:
        ax.legend_.remove()
    plt.savefig(filename, bbox_inches='tight')


def webpanel(result, row, key_descriptions):
    from typing import Tuple, List

    def rmxclabel(d: 'Tuple[str, str, str]',
                  xcs: List) -> 'Tuple[str, str, str]':
        def rm(s: str) -> str:
            for xc in xcs:
                s = s.replace('({})'.format(xc), '')
            return s.rstrip()

        return tuple(rm(s) for s in d)

    panel = {'title': describe_entry('Electronic band structure (PBE)',
                                     panel_description),
             'columns': [
                 [
                     fig('pbe-bs.png', link='pbe-bs.html'),
                 ],
                 [fig('bz-with-gaps.png')]],
             'plot_descriptions': [{'function': bs_pbe,
                                    'filenames': ['pbe-bs.png']},
                                   {'function': bs_pbe_html,
                                    'filenames': ['pbe-bs.html']}],
             'sort': 12}

    return [panel]


@prepare_result
class Result(ASRResult):

    version: int = 0

    bs_soc: dict
    bs_nosoc: dict

    key_descriptions = \
        {
            'bs_soc': 'Bandstructure data with spin-orbit coupling.',
            'bs_nosoc': 'Bandstructure data without spin-orbit coupling.'
        }

    formats = {"ase_webpanel": webpanel}


@command('asr.bandstructure')
@option('-a', '--atoms', help='Atomic structure.',
        type=AtomsFile(), default='structure.json')
@option('-c', '--calculator', help='Calculator params.', type=DictStr())
@option('-b', '--bscalculator',
        help='Bandstructure Calculator params.',
        type=DictStr())
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints',
        type=int,
        help='Number of points along k-point path.')
def main(
        atoms: Atoms,
        calculator: dict = calculate.defaults.calculator,
        bscalculator: dict = calculate.defaults.bscalculator,
        kptpath: Union[str, None] = None,
        npoints: int = 400) -> Result:
    from ase.spectrum.band_structure import get_band_structure
    from ase.dft.kpoints import BandPath
    import copy
    import numpy as np
    from asr.utils.gpw2eigs import gpw2eigs
    from asr.magnetic_anisotropy import get_spin_axis, get_spin_index

    bsrecord = calculate(
        atoms=atoms,
        calculator=calculator,
        bscalculator=bscalculator,
        npoints=npoints,
        kptpath=kptpath,
    )
    gsrecord = calculategs(atoms=atoms, calculator=calculator)
    ref = gsrecord.result.calculation.load().get_fermi_level()
    calc = bsrecord.result.calculation.load()
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
                                            npoints=path['npoints'])
    bs = get_band_structure(calc=calc, path=path, reference=ref)

    results = {}
    bsresults = bs.todict()

    # Save Fermi levels
    maingsrecord = maings(atoms=atoms, calculator=calculator)
    gsresults = maingsrecord.result
    efermi_nosoc = gsresults['gaps_nosoc']['efermi']
    bsresults['efermi'] = efermi_nosoc

    # We copy the bsresults dict because next we will add SOC
    results['bs_nosoc'] = copy.deepcopy(bsresults)  # BS with no SOC

    # Add spin orbit correction
    bsresults = bs.todict()

    theta, phi = get_spin_axis(atoms=atoms, calculator=calculator)

    # We use a larger symmetry tolerance because we want to correctly
    # color spins which doesn't always happen due to slightly broken
    # symmetries, hence tolerance=1e-2.
    # XXX This is only compatible with GPAW
    bsfile = bsrecord.result.calculation.paths[0]
    e_km, _, s_kvm = gpw2eigs(
        bsfile, soc=True, return_spin=True, theta=theta, phi=phi,
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
            get_spin_index(atoms=atoms,
                           calculator=calculator),
            :]  # take x, y or z component
    else:
        sz_mk = s_mvk

    assert sz_mk.shape[1] == npoints, f'sz_mk has wrong dims, {npoints}'

    bsresults['sz_mk'] = sz_mk

    return Result.fromdata(
        bs_soc=singleprec_dict(bsresults),
        bs_nosoc=singleprec_dict(results['bs_nosoc'])
    )


if __name__ == '__main__':
    main.cli()
