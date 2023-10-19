import functools
import warnings
import itertools
import numpy as np
import typing
from typing import List, Tuple, Optional, Dict, Any
from ase.formula import Formula
from ase.db.row import AtomsRow
from ase.phasediagram import PhaseDiagram
from ase.dft.kpoints import BandPath
from ase.dft.kpoints import labels_from_kpts
from asr.core import ASRResult, prepare_result
from asr.database.browser import (
    WebPanel,
    table, matrixtable,
    fig,
    href, dl, code, bold, br, div,
    describe_entry,
    entry_parameter_description,
    make_panel_description)
from asr.utils.hacks import gs_xcname_from_row
from matplotlib import patches

from asr.createwebpanel import (
    OpticalWebpanel, PlasmaWebpanel, InfraredWebpanel, BSEWebpanel,
    BaderWebpanel, BornChargesWebpanel, ChargeNeutralityWebpanel,
    BandstructureWebpanel,
    BerryWebpanel,
)
from asr.extra_result_plotting import (
    gaps_from_row, _absorption,
    create_plot_simple, plot_with_colors, add_bs_ks, legend_on_top,
)


# Optical Panel
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

    @staticmethod
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
# Plasma Panel
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
# Infrared Panel
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

    @staticmethod
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
# BSE
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

    @staticmethod
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


######### Charge Analysis #########
# Bader
@prepare_result
class BaderResult(ASRResult):

    bader_charges: np.ndarray
    sym_a: List[str]

    key_descriptions = {'bader_charges': 'Array of charges [\\|e\\|].',
                        'sym_a': 'Chemical symbols.'}

    formats = {"ase_webpanel": BaderWebpanel}
# born charges
@prepare_result
class BornChargesResult(ASRResult):

    Z_avv: np.ndarray
    sym_a: typing.List[str]

    key_descriptions = {'Z_avv': 'Array of borncharges.',
                        'sym_a': 'Chemical symbols.'}

    formats = {"ase_webpanel": BornChargesWebpanel}
# charge_neutrality
@prepare_result
class ChargeNeutralityResult(ASRResult):
    """Container for asr.charge_neutrality results."""
    from asr.randomresults import SelfConsistentResult
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def plot_formation_scf(row, fname):
        """Plot formation energy diagram and SC Fermi level wrt. VBM."""
        import matplotlib.pyplot as plt
        from asr.extra_result_plotting import (
            plot_lowest_lying, draw_band_edges, draw_ef, set_limits,
            set_labels_and_legend
        )

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



######### Bandstructure #########
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

    @staticmethod
    def plot_bs_html(row, filename='bs.html'):
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

    @staticmethod
    def plot_bs_png(row, filename='bs.png', figsize=(5.5, 5), s=0.5):

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


######## Magnetic Properties ########
# Berry
@prepare_result
class BerryResult(ASRResult):
    Topology: str
    key_descriptions = {'Topology': 'Band topology.'}
    formats = {"ase_webpanel": BerryWebpanel}

    @staticmethod
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


######### defectformation #########
# skip because it doesnt make a webpanel


######### defect info #########
def DefectInfoWebpanel(result, row, key_descriptions):
    spglib = href('SpgLib', 'https://spglib.github.io/spglib/')
    crystal_type = describe_crystaltype_entry(spglib)

    spg_list_link = href(
        'space group', 'https://en.wikipedia.org/wiki/List_of_space_groups')
    spacegroup = describe_entry(
        'Space group',
        f"The {spg_list_link} is determined with {spglib}.")
    pointgroup = describe_pointgroup_entry(spglib)
    host_hof = describe_entry(
        'Heat of formation',
        result.key_descriptions['host_hof'])
    # XXX get correct XC name
    host_gap_pbe = describe_entry(
        'PBE band gap',
        'PBE band gap of the host crystal [eV].')
    host_gap_hse = describe_entry(
        'HSE band gap',
        'HSE band gap of the host crystal [eV].')
    R_nn = describe_entry(
        'Defect-defect distance',
        result.key_descriptions['R_nn'])

    # extract defect name, charge state, and format it
    defect_name = row.defect_name
    if defect_name != 'pristine':
        defect_name = (f'{defect_name.split("_")[0]}<sub>{defect_name.split("_")[1]}'
                       '</sub>')
        charge_state = row.charge_state
        q = charge_state.split()[-1].split(')')[0]

    # only show results for the concentration if charge neutrality results present
    show_conc = 'results-asr.charge_neutrality.json' in row.data
    if show_conc and defect_name != 'pristine':
        conc_res = row.data['results-asr.charge_neutrality.json']
        conc_row = get_concentration_row(conc_res, defect_name, q)

    uid = result.host_uid
    uidstring = describe_entry(
        'C2DB link',
        'Link to C2DB entry of the host material.')

    # define overview table with described entries and corresponding results
    lines = [[crystal_type, result.host_crystal],
             [spacegroup, result.host_spacegroup],
             [pointgroup, result.host_pointgroup],
             [host_hof, f'{result.host_hof:.2f} eV/atom'],
             [host_gap_pbe, f'{result.host_gap_pbe:.2f} eV']]
    basictable = table(result, 'Pristine crystal', [])
    basictable['rows'].extend(lines)

    # add additional data to the table if HSE gap, defect-defect distance,
    # concentration, and host uid are present
    if result.host_gap_hse is not None:
        basictable['rows'].extend(
            [[host_gap_hse, f'{result.host_gap_hse:.2f} eV']])
    defecttable = table(result, 'Defect properties', [])
    if result.R_nn is not None:
        defecttable['rows'].extend(
            [[R_nn, f'{result.R_nn:.2f} Å']])
    if show_conc and defect_name != 'pristine':
        defecttable['rows'].extend(conc_row)
    if uid:
        basictable['rows'].extend(
            [[uidstring,
              '<a href="https://cmrdb.fysik.dtu.dk/c2db/row/{uid}"'
              '>{uid}</a>'.format(uid=uid)]])

    panel = {'title': 'Summary',
             'columns': [[basictable, defecttable], []],
             'sort': -1}

    return [panel]


@prepare_result
class DefectInfoResult(ASRResult):
    """Container for asr.defectinfo results."""

    defect_name: str
    host_name: str
    charge_state: str
    host_pointgroup: str
    host_spacegroup: str
    host_crystal: str
    host_uid: str
    host_hof: float
    host_gap_pbe: float
    host_gap_hse: float
    R_nn: float

    key_descriptions: typing.Dict[str, str] = dict(
        defect_name='Name of the defect({type}_{position}).',
        host_name='Name of the host system.',
        charge_state='Charge state of the defect system.',
        host_pointgroup='Point group of the host crystal.',
        host_spacegroup='Space group of the host crystal.',
        host_crystal='Crystal type of the host crystal.',
        host_uid='UID of the primitive host crystal.',
        host_hof='Heat of formation for the host crystal [eV/atom].',
        host_gap_pbe='PBE bandgap of the host crystal [eV].',
        host_gap_hse='HSE bandgap of the host crystal [eV].',
        R_nn='Nearest neighbor distance of repeated defects [Å].')

    formats = {"ase_webpanel": DefectInfoWebpanel}


def get_concentration_row(conc_res, defect_name, q):
    rowlist = []
    for scresult in conc_res.scresults:
        condition = scresult.condition
        for i, element in enumerate(scresult['defect_concentrations']):
            conc_row = describe_entry(
                f'Eq. concentration ({condition})',
                'Equilibrium concentration at self-consistent Fermi level.')
            if element['defect_name'] == defect_name:
                for altel in element['concentrations']:
                    if altel[1] == int(q):
                        concentration = altel[0]
                        rowlist.append([conc_row,
                                        f'{concentration:.1e} cm<sup>-2</sup>'])

    return rowlist


######### DefectLinks #########
def DefectLinksWebpanel(result, row, key_description):
    baselink = 'https://cmrdb.fysik.dtu.dk/qpod/row/'

    # initialize table for charged and neutral systems
    charged_table = table(row, 'Other charge states', [])
    neutral_table = table(row, 'Other defects', [])
    # fill in values for the two tables from the result object
    charged_table = extend_table(charged_table, result, 'charged', baselink)
    neutral_table = extend_table(neutral_table, result, 'neutral', baselink)
    neutral_table = extend_table(neutral_table, result, 'pristine', baselink)

    # define webpanel
    panel = WebPanel('Other defects',
                     columns=[[charged_table], [neutral_table]],
                     sort=45)

    return [panel]


def extend_table(table, result, resulttype, baselink):
    if resulttype == 'pristine':
        tmpresult = result.pristinelinks
    elif resulttype == 'neutral':
        tmpresult = result.neutrallinks
    elif resulttype == 'charged':
        tmpresult = result.chargedlinks
    else:
        raise RuntimeError('did not find {resulttype} results!')

    for element in tmpresult:
        if element[1].startswith('V'):
            linkstring = element[1].replace('V', 'v', 1)
        else:
            linkstring = element[1]
        table['rows'].extend(
            [[f'{linkstring}',
              f'<a href="{baselink}{element[0]}">link</a>']])

    return table


@prepare_result
class DefectLinksResult(ASRResult):
    """Container for defectlinks results."""

    chargedlinks: typing.List
    neutrallinks: typing.List
    pristinelinks: typing.List

    key_descriptions = dict(
        chargedlinks='Links tuple for the charged states of the same defect.',
        neutrallinks='Links tuple for other defects within the same material.',
        pristinelinks='Link tuple for pristine material.')

    formats = {'ase_webpanel': DefectLinksWebpanel}


######### deformationpotentials #########
description_text = """\
The deformation potentials represent the energy shifts of the
bottom of the conduction band (CB) and the top of the valence band
(VB) at a given k-point, under an applied strain.

The two tables at the top show the deformation potentials for the
valence band (D<sub>VB</sub>) and conduction band (D<sub>CB</sub>)
at the high-symmetry k-points, subdivided into the different strain
components. At the bottom of each table are shown the
deformation potentials at the k-points where the VBM and CBM are found
(k<sub>VBM</sub> and k<sub>CBM</sub>, respectively).
Note that the latter may coincide with any of the high-symmetry k-points.
The table at the bottom shows the band gap deformation potentials.

All the values shown are calculated with spin-orbit coupling (SOC).
Values obtained without SOC can be found in the material raw data.
"""


panel_description = make_panel_description(
    description_text,
    articles=[
        href("""Wiktor, J. and Pasquarello, A., 2016. Absolute deformation potentials
of two-dimensional materials. Physical Review B, 94(24), p.245411""",
             "https://doi.org/10.1103/PhysRevB.94.245411")
    ],
)



def get_table_row(kpt, band, data):
    row = []
    for comp in ['xx', 'yy', 'xy']:
        row.append(data[kpt][comp][band])
    return np.asarray(row)


def DefPotsWebpanel(result, row, key_descriptions):
    from asr.database.browser import matrixtable, describe_entry, WebPanel

    def get_basename(kpt):
        if kpt == 'G':
            return 'Γ'
        elif kpt in ('VBM', 'CBM'):
            return f'k<sub>{kpt}</sub>'
        else:
            return kpt

    description = describe_entry('Deformation potentials', panel_description)
    defpots = result['deformation_potentials_soc'].copy()  # change this back
    # to defpots_soc
    columnlabels = ['xx', 'yy', 'xy']

    dp_gap = defpots.pop('gap')
    dp_list_vb = []
    dp_list_cb = []
    add_to_bottom_vb = []
    add_to_bottom_cb = []
    dp_labels_cb = []
    dp_labels_vb = []

    for kpt in defpots:
        dp_labels = []
        label = get_basename(kpt)
        for band, table, bottom, lab in zip(
                ['VB', 'CB'],
                [dp_list_vb, dp_list_cb],
                [add_to_bottom_vb, add_to_bottom_cb],
                [dp_labels_vb, dp_labels_cb]):
            row = get_table_row(kpt, band, defpots)
            if 'k' in label:
                if band in label:
                    bottom.append((label, row))
                    continue
                else:
                    continue
            lab.append(label)
            table.append(row)

    for label, row in add_to_bottom_vb:
        dp_list_vb.append(row)
        dp_labels_vb.append(label)
    for label, row in add_to_bottom_cb:
        dp_list_cb.append(row)
        dp_labels_cb.append(label)

    dp_labels.append('Band Gap')
    dp_list_gap = [[dp_gap[comp] for comp in ['xx', 'yy', 'xy']]]

    dp_table_vb = matrixtable(
        dp_list_vb,
        digits=2,
        title='D<sub>VB</sub> (eV)',
        columnlabels=columnlabels,
        rowlabels=dp_labels_vb
    )
    dp_table_cb = matrixtable(
        dp_list_cb,
        digits=2,
        title='D<sub>CB</sub> (eV)',
        columnlabels=columnlabels,
        rowlabels=dp_labels_cb
    )
    dp_table_gap = matrixtable(
        dp_list_gap,
        digits=2,
        title='',
        columnlabels=columnlabels,
        rowlabels=['Band Gap']
    )
    panel = WebPanel(
        description,
        columns=[[dp_table_vb, dp_table_gap], [dp_table_cb]],
        sort=4
    )
    return [panel]


@prepare_result
class DefPotsResult(ASRResult):

    defpots_nosoc: typing.Dict[str, float]
    defpots_soc: typing.Dict[str, float]
    kpts_defpots_nosoc: typing.Union[list, typing.Dict[str, float]]
    kpts_defpots_soc: typing.Union[list, typing.Dict[str, float]]

    key_descriptions = {
        'defpots_nosoc': (
            'Deformation potentials under different types of '
            'deformations (xx, yy, zz, yz, xz, xy) at each k-point, '
            'without SOC'),
        'defpots_soc': (
            'Deformation potentials under different applied strains '
            '(xx, yy, zz, yz, xz, xy) at each k-point, with SOC'),
        'kpts_defpots_nosoc': (
            'k-points at which deformation potentials were calculated '
            'without spin-orbit coupling'),
        'kpts_defpots_soc': (
            'k-points at which deformation potentials were calculated '
            'with spin-orbit coupling'),
    }

    formats = {"ase_webpanel": DefPotsWebpanel}


######### dimensionality #########
def get_dimtypes():
    """Create a list of all dimensionality types."""
    from itertools import product
    s = set(product([0, 1], repeat=4))
    s2 = sorted(s, key=lambda x: (sum(x), *[-t for t in x]))[1:]
    string = "0123"
    return ["".join(x for x, y in zip(string, s3) if y) + "D" for s3 in s2]


def DimWebpanel(result, row, key_descriptions):
    from asr.database.browser import table, fig
    dimtable = table(row, 'Dimensionality scores',
                     [f'dim_score_{dimtype}' for dimtype in get_dimtypes()],
                     key_descriptions, 2)
    panel = {'title': 'Dimensionality analysis',
             'columns': [[dimtable], [fig('dimensionality-histogram.png')]]}
    return [panel]


######### dos #########
panel_description = make_panel_description(
    """Density of States
""")


def DOSWebpanel(result: ASRResult, row, key_descriptions: dict) -> list:
    parameter_description = entry_parameter_description(
        row.data,
        'asr.dos')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Density of States',
                                     description=title_description),
             'columns': [[fig('dos.png')]],
             'plot_descriptions':
                 [{'function': dos_plot,
                   'filenames': ['dos.png']}]}

    return [panel]


@prepare_result
class DOSResult(ASRResult):
    dosspin0_e: List[float]
    dosspin1_e: List[float]
    energies_e: List[float]
    natoms: int
    volume: float

    key_descriptions = {'dosspin0_e': 'Spin up DOS [states/eV]',
                        'dosspin1_e': 'Spin up DOS [states/eV]',
                        'energies_e': 'Energies relative to Fermi level [eV]',
                        'natoms': 'Number of atoms',
                        'volume': 'Volume of unit cell [Ang^3]'}
    formats = {"ase_webpanel": DOSWebpanel}


Result = DOSResult  # backwards compatibility with old result files


def dos_plot(row, filename: str):
    import matplotlib.pyplot as plt
    dos = row.data.get('results-asr.dos.json')
    x = dos['energies_e']
    y0 = dos['dosspin0_e']
    y1 = dos.get('dosspin1_e')
    fig, ax = plt.subplots()
    if y1:
        ax.plot(x, y0, label='up')
        ax.plot(x, y1, label='down')
        ax.legend()
    else:
        ax.plot(x, y0)

    ax.set_xlabel(r'Energy - $E_\mathrm{F}$ [eV]')
    ax.set_ylabel('DOS [electrons/eV]')
    fig.tight_layout()
    fig.savefig(filename)
    return [ax]


######### Emasses #########
MAXMASS = 10  # More that 90% of masses are less than this
# This mass is only used to limit bandstructure plots


panel_description = make_panel_description(
    """
The effective mass tensor represents the second derivative of the band energy
w.r.t. wave vector at a band extremum. The effective masses of the valence
bands (VB) and conduction bands (CB) are obtained as the eigenvalues of the
mass tensor. The latter is determined by fitting a 2nd order polynomium to the
band energies on a fine k-point mesh around the band extrema. Spin–orbit
interactions are included. The fit curve is shown for the highest VB and
lowest CB. The “parabolicity” of the band is quantified by the
mean absolute relative error (MARE) of the fit to the band energy in an energy
range of 25 meV.
""",
    articles=[
        'C2DB',
    ],
)


def mareformat(mare):
    return str(round(mare, 3)) + " %"


def get_emass_dict_from_row(row, has_mae=False):
    import numpy as np
    if has_mae:
        results = row.data['results-asr.emasses@validate.json']
    else:
        results = row.data.get('results-asr.emasses.json')

    cb_indices = []
    vb_indices = []
    for k in results.keys():
        if '(' in k and ')' in k:
            for k2 in results[k].keys():
                if 'nosoc' in k2:
                    break

                if 'vb_soc_effmass' in k2:
                    vb_indices.append(k)
                    break
                elif 'cb_soc_effmass' in k2:
                    cb_indices.append(k)
                    break

    cb_indices = [(k, convert_key_to_tuple(k)[1]) for k in cb_indices]
    vb_indices = [(k, convert_key_to_tuple(k)[1]) for k in vb_indices]

    ordered_cb_indices = sorted(cb_indices, key=lambda el: el[1])
    ordered_vb_indices = sorted(vb_indices, key=lambda el: -el[1])

    def get_the_dict(ordered_indices, name, offset_sym, has_mae):
        # Write a dictionary that will be turned into a table
        # The dict keys are the table row name
        # and the dict values are the effective masses
        # key: name offset_sym(bol) offset_num direction i
        # E.g. key: VB -2 direction 2
        # value: <number> m_e
        # E.g. value: 0.41 m_e
        my_dict = {}
        for offset_num, (key, band_number) in enumerate(ordered_indices):
            data = results[key]
            direction = 0
            marekey = name.lower() + '_soc_wideareaPARAMARE'
            if marekey not in data:
                print(f'WARNING: Your data is outdated. Please rerun emasses@validate.')
                mares = None
                has_mae = False
            else:
                mares = data[marekey] if has_mae else None

            for k in data.keys():
                if 'effmass' in k:
                    mass = data[k]
                    if mass is not None and not np.isnan(mass):
                        direction += 1
                        expectedsign = 1 if name == "CB" else -1
                        if abs(mass) > 3000 or np.sign(mass) != expectedsign:
                            mass_str = "N/A"
                        else:
                            mass_str = str(round(abs(mass) * 100)
                                           / 100) + " m<sub>0</sub>"

                        if has_mae:
                            mare = mares[direction - 1]
                            marestr = mareformat(mare)

                            if offset_num == 0:
                                my_dict[f'{name}, direction {direction}'] = \
                                    (f'{mass_str}', marestr)
                            else:
                                my_dict['{} {} {}, direction {}'.format(
                                    name, offset_sym,
                                    offset_num, direction)] = \
                                    (f'{mass_str}', marestr)

                        else:
                            if offset_num == 0:
                                my_dict[f'{name}, direction {direction}'] = \
                                    f'{mass_str}'
                            else:
                                my_dict['{} {} {}, direction {}'.format(
                                    name, offset_sym,
                                    offset_num, direction)] = \
                                    f'{mass_str}'

        return my_dict

    electron_dict = get_the_dict(ordered_cb_indices, 'CB', '+', has_mae)
    hole_dict = get_the_dict(ordered_vb_indices, 'VB', '-', has_mae)

    return electron_dict, hole_dict


def custom_table(values_dict, title, has_mae=False):
    rows = []
    for k in values_dict.keys():
        if has_mae:
            rows.append((k, values_dict[k][0], values_dict[k][1]))
        else:
            rows.append((k, values_dict[k]))

    if has_mae:
        table = {'type': 'table',
                 'header': [title, 'Value', 'MARE (25 meV)']}
    else:
        table = {'type': 'table',
                 'header': [title, 'Value']}

    table['rows'] = rows
    return table


def create_columns_fnames(row):
    from asr.database.browser import fig as asrfig

    results = row.data.get('results-asr.emasses.json')

    cb_fnames = []
    vb_fnames = []

    vb_indices = []
    cb_indices = []

    for spin_band_str, data in results.items():
        if '__' in spin_band_str or not isinstance(data, dict):
            continue
        is_w_soc = check_soc(data)
        if not is_w_soc:
            continue
        for k in data.keys():
            if 'effmass' in k and 'vb' in k:
                vb_indices.append(spin_band_str)
                break
            if 'effmass' in k and 'cb' in k:
                cb_indices.append(spin_band_str)
                break

    cb_masses = {}
    vb_masses = {}

    for cb_key in cb_indices:
        data = results[cb_key]
        masses = []
        for k in data.keys():
            if 'effmass' in k:
                masses.append(data[k])
        tuple_key = convert_key_to_tuple(cb_key)
        cb_masses[tuple_key] = masses

    for vb_key in vb_indices:
        data = results[vb_key]
        masses = []
        for k in data.keys():
            if 'effmass' in k:
                masses.append(data[k])
        tuple_key = convert_key_to_tuple(vb_key)
        vb_masses[tuple_key] = masses

    for direction in range(3):
        should_plot = True
        for cb_key in cb_indices:
            data = results[cb_key]
            fit_data_list = data['cb_soc_bzcuts']
            if direction >= len(fit_data_list):
                should_plot = False
                continue
        if should_plot:
            fname = 'cb_dir_{}.png'.format(direction)
            cb_fnames.append(fname)

            fname = 'vb_dir_{}.png'.format(direction)
            vb_fnames.append(fname)

    assert len(cb_fnames) == len(vb_fnames), \
        'Num cb plots: {}\nNum vb plots: {}'.format(
        len(cb_fnames), len(vb_fnames))

    num_figs = len(cb_fnames)

    columns = [[], []]
    for j in range(num_figs):
        cb_fname = cb_fnames[j]
        vb_fname = vb_fnames[j]

        columns[0].append(asrfig(cb_fname))
        columns[1].append(asrfig(vb_fname))

    return columns, cb_fnames + vb_fnames


def EmassesWebpanel(result, row, key_descriptions):
    has_mae = 'results-asr.emasses@validate.json' in row.data
    columns, fnames = create_columns_fnames(row)

    electron_dict, hole_dict = get_emass_dict_from_row(row, has_mae)

    electron_table = custom_table(electron_dict, 'Electron effective mass', has_mae)
    hole_table = custom_table(hole_dict, 'Hole effective mass', has_mae)
    columns[0].append(electron_table)
    columns[1].append(hole_table)

    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)

    panel = {'title': describe_entry(f'Effective masses ({xcname})',
                                     panel_description),
             'columns': columns,
             'plot_descriptions':
             [{'function': make_the_plots,
               'filenames': fnames
               }],
             'sort': 14}
    return [panel]


class EmassesResult(ASRResult):
    pass


@prepare_result
class ValidateResult(ASRResult):

    formats = {"ase_webpanel": EmassesWebpanel}


def convert_key_to_tuple(key):
    k = key.replace("(", "").replace(")", "")
    ks = k.split(",")
    ks = [k.strip() for k in ks]
    ks = [int(k) for k in ks]
    return tuple(ks)


def make_the_plots(row, *args):
    # Loop through directions, each direction is a column
    # For direction i, loop through cbs and plot on fig
    # -- Plot also quadratic fit from curvature/effective mass value
    # For direction i, loop through vbs and plot on fig
    # Make a final column containing a table with the numerical values
    # for the effective masses
    import matplotlib.pyplot as plt
    from asr.database.browser import fig as asrfig

    results = row.data.get('results-asr.emasses.json')
    efermi = row.efermi
    sdir = row.get('spin_axis', 'z')
    cell_cv = row.cell

    reference = row.get('evac', efermi)

    # Check whether material necessarily has no spin-degeneracy
    spin_degenerate = row.magstate == 'NM' and row.has_inversion_symmetry

    label = r'E_\mathrm{vac}' if 'evac' in row else r'E_\mathrm{F}'
    columns = []
    cb_fnames = []
    vb_fnames = []

    vb_indices = []
    cb_indices = []

    for spin_band_str, data in results.items():
        if '__' in spin_band_str or not isinstance(data, dict):
            continue
        is_w_soc = check_soc(data)
        if not is_w_soc:
            continue
        for k in data.keys():
            if 'effmass' in k and 'vb' in k:
                vb_indices.append(spin_band_str)
                break
            if 'effmass' in k and 'cb' in k:
                cb_indices.append(spin_band_str)
                break

    cb_masses = {}
    vb_masses = {}

    for cb_key in cb_indices:
        data = results[cb_key]
        masses = []
        for k in data.keys():
            if 'effmass' in k:
                masses.append(data[k])
        tuple_key = convert_key_to_tuple(cb_key)
        cb_masses[tuple_key] = masses

    for vb_key in vb_indices:
        data = results[vb_key]
        masses = []
        for k in data.keys():
            if 'effmass' in k:
                masses.append(data[k])
        tuple_key = convert_key_to_tuple(vb_key)
        vb_masses[tuple_key] = masses

    plt_count = 0
    for direction in range(3):
        # CB plots
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 2.8),
                                 sharey=True,
                                 gridspec_kw={'width_ratios': [1]})

        should_plot = True
        for i, cb_key in enumerate(cb_indices):
            cb_tuple = convert_key_to_tuple(cb_key)
            data = results[cb_key]
            fit_data_list = data['cb_soc_bzcuts']

            # Hacky handling of different dimensionalities
            if direction >= len(fit_data_list):
                should_plot = False
                continue

            mass = cb_masses[cb_tuple][direction]
            fit_data = fit_data_list[direction]

            if i == 0:
                kpts_kv, xk, e_km, sz_km = get_plot_data(fit_data, reference,
                                                         cell_cv)
            plot_fit(axes, mass, reference, cell_cv,
                     xk, kpts_kv, data['cb_soc_2ndOrderFit'])

            if i != 0:
                continue

            plot_band(fig, axes, mass, reference, cell_cv,
                      xk, kpts_kv, e_km, sz_km,
                      cbarlabel=rf'$\langle S_{sdir} \rangle$',
                      xlabel=r'$\Delta k$ [1/$\mathrm{\AA}$]',
                      ylabel=r'$E-{}$ [eV]'.format(label),
                      title=f'CB, direction {direction + 1}',
                      bandtype='cb',
                      adjust_view=True, spin_degenerate=spin_degenerate)

        if should_plot:
            fname = args[plt_count]
            plt.savefig(fname)
        plt.close()

        # VB plots
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 2.8),
                                 sharey=True,
                                 gridspec_kw={'width_ratios': [1]})

        for i, vb_key in enumerate(vb_indices):
            vb_tuple = convert_key_to_tuple(vb_key)
            data = results[vb_key]
            fit_data_list = data['vb_soc_bzcuts']
            if direction >= len(fit_data_list):
                continue

            mass = vb_masses[vb_tuple][direction]
            fit_data = fit_data_list[direction]

            if i == 0:
                kpts_kv, xk, e_km, sz_km = get_plot_data(fit_data, reference,
                                                         cell_cv)

            plot_fit(axes, mass, reference, cell_cv,
                     xk, kpts_kv, data['vb_soc_2ndOrderFit'])

            if i != 0:
                continue

            plot_band(fig, axes, mass, reference, cell_cv,
                      xk, kpts_kv, e_km, sz_km,
                      cbarlabel=rf'$\langle S_{sdir} \rangle$',
                      xlabel=r'$\Delta k$ [1/$\mathrm{\AA}$]',
                      ylabel=r'$E-{}$ [eV]'.format(label),
                      title=f'VB, direction {direction + 1}',
                      bandtype='vb',
                      adjust_view=True, spin_degenerate=spin_degenerate)

        if should_plot:
            nplts = len(args)
            fname = args[plt_count + nplts // 2]
            plt.savefig(fname)
        plt.close()

        plt_count += 1

    assert len(cb_fnames) == len(vb_fnames), \
        'Num cb plots: {}\nNum vb plots: {}'.format(
        len(cb_fnames), len(vb_fnames))

    num_cols = len(cb_fnames)

    for j in range(num_cols):
        cb_fname = cb_fnames[j]
        vb_fname = vb_fnames[j]
        col = [asrfig(cb_fname), asrfig(vb_fname)]
        columns.append(col)

    return


def model(kpts_kv):
    """Calculate simple third order model.

    Parameters
    ----------
        kpts_kv: (nk, 3)-shape ndarray
            units of (1 / Bohr)

    """
    import numpy as np
    k_kx, k_ky, k_kz = kpts_kv[:, 0], kpts_kv[:, 1], kpts_kv[:, 2]

    ones = np.ones(len(k_kx))

    A_dp = np.array([k_kx**2,
                     k_ky**2,
                     k_kz**2,
                     k_kx * k_ky,
                     k_kx * k_kz,
                     k_ky * k_kz,
                     k_kx,
                     k_ky,
                     k_kz,
                     ones,
                     k_kx**3,
                     k_ky**3,
                     k_kz**3,
                     k_kx**2 * k_ky,
                     k_kx**2 * k_kz,
                     k_ky**2 * k_kx,
                     k_ky**2 * k_kz,
                     k_kz**2 * k_kx,
                     k_kz**2 * k_ky,
                     k_kx * k_ky * k_kz]).T

    return A_dp


def evalmodel(kpts_kv, c_p, thirdorder=True):
    import numpy as np
    kpts_kv = np.asarray(kpts_kv)
    if kpts_kv.ndim == 1:
        kpts_kv = kpts_kv[np.newaxis]
    A_kp = model(kpts_kv)
    if not thirdorder:
        A_kp = A_kp[:, :10]
    return np.dot(A_kp, c_p)


def get_range(mass, _erange):
    from ase.units import Ha, Bohr
    return (2 * mass * _erange / Ha) ** 0.5 / Bohr


def plot_fit(axes, mass, reference, cell_cv,
             xk2, kpts_kv, fit_coeffs):
    from ase.units import Ha
    emodel_k = evalmodel(kpts_kv,
                         fit_coeffs,
                         thirdorder=False) * Ha - reference
    axes.plot(xk2, emodel_k, c='r', ls='--')


def plot_band(fig, axes, mass, reference, cell_cv,
              xk2, kpts_kv, e_km, sz_km,
              cbarlabel, xlabel, ylabel, title,
              bandtype,
              adjust_view=True, spin_degenerate=False):
    import matplotlib.pyplot as plt
    shape = e_km.shape
    perm = (-sz_km).argsort(axis=None)
    repeated_xcoords = np.vstack([xk2] * shape[1]).T
    flat_energies = e_km.ravel()[perm]
    flat_xcoords = repeated_xcoords.ravel()[perm]

    if spin_degenerate:
        colors = np.zeros_like(flat_energies)
    else:
        colors = sz_km.ravel()[perm]

    scatterdata = axes.scatter(flat_xcoords, flat_energies,
                               c=colors, vmin=-1, vmax=1)

    if adjust_view:
        erange = 0.05  # 50 meV
        if bandtype == 'cb':
            y1 = np.min(e_km[:, -1]) - erange * 0.25
            y2 = np.min(e_km[:, -1]) + erange * 0.75
        else:
            y1 = np.max(e_km[:, -1]) - erange * 0.75
            y2 = np.max(e_km[:, -1]) + erange * 0.25
        axes.set_ylim(y1, y2)

        my_range = get_range(min(MAXMASS, abs(mass)), erange)
        axes.set_xlim(-my_range, my_range)

        cbar = fig.colorbar(scatterdata, ax=axes)
        cbar.set_label(cbarlabel)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.update_ticks()
    plt.locator_params(axis='x', nbins=3)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    plt.tight_layout()


def get_plot_data(fit_data, reference, cell_cv):
    from ase.units import Bohr
    from ase.dft.kpoints import kpoint_convert, labels_from_kpts
    ks = fit_data['kpts_kc']
    e_km = fit_data['e_km'] - reference
    sz_km = fit_data['spin_km']
    xk, y, y2 = labels_from_kpts(kpts=ks, cell=cell_cv, eps=1)
    xk -= xk[-1] / 2

    kpts_kv = kpoint_convert(cell_cv=cell_cv, skpts_kc=ks)
    kpts_kv *= Bohr

    return kpts_kv, xk, e_km, sz_km


def check_soc(spin_band_dict):
    for k in spin_band_dict.keys():
        if 'effmass' in k and 'nosoc' in k:
            return False

    return True


######### exchange #########
def ExchangeWebpanel(result, row, key_descriptions):
    from asr.database.browser import (table,
                                      entry_parameter_description,
                                      describe_entry, WebPanel)
    if row.get('magstate', 'NM') == 'NM':
        return []

    parameter_description = entry_parameter_description(
        row.data,
        'asr.exchange@calculate')
    explanation_J = ('The nearest neighbor exchange coupling\n\n'
                     + parameter_description)
    explanation_lam = ('The nearest neighbor isotropic exchange coupling\n\n'
                       + parameter_description)
    explanation_A = ('The single ion anisotropy\n\n'
                     + parameter_description)
    explanation_spin = ('The spin of magnetic atoms\n\n'
                        + parameter_description)
    explanation_N = ('The number of nearest neighbors\n\n'
                     + parameter_description)
    J = describe_entry('J', description=explanation_J)
    lam = describe_entry('lam', description=explanation_lam)
    A = describe_entry('A', description=explanation_A)
    spin = describe_entry('spin', description=explanation_spin)
    N_nn = describe_entry('N_nn', description=explanation_N)

    heisenberg_table = table(row, 'Heisenberg model',
                             [J, lam, A, spin, N_nn],
                             kd=key_descriptions)
    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)
    panel = WebPanel(title=f'Basic magnetic properties ({xcname})',
                     columns=[[heisenberg_table], []],
                     sort=11)
    return [panel]


@prepare_result
class ExchangeResult(ASRResult):

    J: float
    A: float
    lam: float
    spin: float
    N_nn: int

    key_descriptions = {
        'J': "Nearest neighbor exchange coupling [meV]",
        'A': "Single-ion anisotropy (out-of-plane) [meV]",
        'lam': "Anisotropic exchange (out-of-plane) [meV]",
        'spin': "Maximum value of S_z at magnetic sites",
        'N_nn': "Number of nearest neighbors",
    }

    formats = {"ase_webpanel": ExchangeWebpanel}


######### fermisurface #########
panel_description = make_panel_description(
    """The Fermi surface calculated with spin–orbit interactions. The expectation
value of S_i (where i=z for non-magnetic materials and otherwise is the
magnetic easy axis) indicated by the color code.""",
    articles=[
        'C2DB',
    ],
)


def plot_fermi(row, fname, sfs=1, dpi=200):
    from ase.geometry.cell import Cell
    from matplotlib import pyplot as plt
    from asr.utils.symmetry import c2db_symmetry_eps
    cell = Cell(row.cell)
    lat = cell.get_bravais_lattice(pbc=row.pbc, eps=c2db_symmetry_eps)
    plt.figure(figsize=(5, 4))
    ax = lat.plot_bz(vectors=False, pointstyle={'c': 'k', 'marker': '.'})
    add_fermi(row, ax=ax, s=sfs)
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)


def add_fermi(row, ax, s=0.25):
    from matplotlib import pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    verts = row.data['results-asr.fermisurface.json']['contours'].copy()
    normalize = colors.Normalize(vmin=-1, vmax=1)
    verts[:, :2] /= (2 * np.pi)
    im = ax.scatter(verts[:, 0], verts[:, 1], c=verts[:, -1],
                    s=s, cmap='viridis', marker=',',
                    norm=normalize, alpha=1, zorder=2)

    sdir = row.get('spin_axis', 'z')
    cbar = plt.colorbar(im, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params()
    cbar.set_label(r'$\langle S_{} \rangle $'.format(sdir))


def FermiWebpanel(result, row, key_descriptions):

    panel = {'title': describe_entry('Fermi surface', panel_description),
             'columns': [[fig('fermi_surface.png')]],
             'plot_descriptions': [{'function': plot_fermi,
                                    'filenames': ['fermi_surface.png']}],
             'sort': 13}

    return [panel]


@prepare_result
class FermiSurfaceResult(ASRResult):

    contours: list
    key_descriptions = {'contours': 'List of Fermi surface contours.'}

    formats = {"ase_webpanel": FermiWebpanel}


######### get_wfs #########
@prepare_result
class WaveFunctionResult(ASRResult):
    """Container for results of specific wavefunction for one spin channel."""

    state: int
    spin: int
    energy: float

    key_descriptions: typing.Dict[str, str] = dict(
        state='State index.',
        spin='Spin index (0 or 1).',
        energy='Energy of the state (ref. to the vacuum level in 2D) [eV].')


@prepare_result
class WfsResult(ASRResult):
    """Container for asr.get_wfs results."""

    wfs: typing.List[WaveFunctionResult]
    above_below: typing.Tuple[bool, bool]
    eref: float

    key_descriptions: typing.Dict[str, str] = dict(
        wfs='List of WaveFunctionResult objects for all states.',
        above_below='States within the gap above and below EF? '
                    '(ONLY for defect systems).',
        eref='Energy reference (vacuum level in 2D, 0 otherwise) [eV].')


######### gs #########
panel_description = make_panel_description(
    """
Electronic properties derived from a ground state density functional theory
calculation.
""",
    articles=['C2DB'],
)


def _explain_bandgap(row, gap_name):
    parameter_description = _get_parameter_description(row)

    if gap_name == 'gap':
        name = 'Band gap'
        adjective = ''
    elif gap_name == 'gap_dir':
        name = 'Direct band gap'
        adjective = 'direct '
    else:
        raise ValueError(f'Bad gapname {gap_name}')

    txt = (f'The {adjective}electronic single-particle band gap '
           'including spin–orbit effects.')

    description = f'{txt}\n\n{parameter_description}'
    return describe_entry(name, description=description)


def vbm_or_cbm_row(title, quantity_name, reference_explanation, value):
    description = (f'Energy of the {quantity_name} relative to the '
                   f'{reference_explanation}. '
                   'Spin–orbit coupling is included.')
    return [describe_entry(title, description=description), f'{value:.2f} eV']


def _get_parameter_description(row):
    desc = entry_parameter_description(
        row.data,
        'asr.gs@calculate',
        exclude_keys=set(['txt', 'fixdensity', 'verbose', 'symmetry',
                          'idiotproof', 'maxiter', 'hund', 'random',
                          'experimental', 'basis', 'setups']))
    return desc


def bz_with_band_extremums(row, fname):
    from ase.geometry.cell import Cell
    from matplotlib import pyplot as plt
    import numpy as np
    from asr.utils.symmetry import c2db_symmetry_eps

    ndim = sum(row.pbc)

    # Standardize the cell rotation via Bravais lattice roundtrip:
    lat = Cell(row.cell).get_bravais_lattice(pbc=row.pbc,
                                             eps=c2db_symmetry_eps)
    cell = lat.tocell()

    plt.figure(figsize=(4, 4))
    lat.plot_bz(vectors=False, pointstyle={'c': 'k', 'marker': '.'})

    gsresults = row.data.get('results-asr.gs.json')
    cbm_c = gsresults['k_cbm_c']
    vbm_c = gsresults['k_vbm_c']
    op_scc = row.data[
        'results-asr.structureinfo.json']['spglib_dataset']['rotations']
    if cbm_c is not None:
        if not row.is_magnetic:
            op_scc = np.concatenate([op_scc, -op_scc])
        ax = plt.gca()
        icell_cv = cell.reciprocal()
        vbm_style = {'marker': 'o', 'facecolor': 'w',
                     'edgecolors': 'C0', 's': 50, 'lw': 2,
                     'zorder': 4}
        cbm_style = {'c': 'C1', 'marker': 'o', 's': 20, 'zorder': 5}
        cbm_sc = np.dot(op_scc.transpose(0, 2, 1), cbm_c)
        vbm_sc = np.dot(op_scc.transpose(0, 2, 1), vbm_c)
        cbm_sv = np.dot(cbm_sc, icell_cv)
        vbm_sv = np.dot(vbm_sc, icell_cv)

        if ndim < 3:
            ax.scatter([vbm_sv[:, 0]], [vbm_sv[:, 1]], **vbm_style, label='VBM')
            ax.scatter([cbm_sv[:, 0]], [cbm_sv[:, 1]], **cbm_style, label='CBM')

            # We need to keep the limits set by ASE in 3D, else the aspect
            # ratio goes haywire.  Hence this bit is also for ndim < 3 only.
            xlim = np.array(ax.get_xlim()) * 1.4
            ylim = np.array(ax.get_ylim()) * 1.4
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ax.scatter([vbm_sv[:, 0]], [vbm_sv[:, 1]],
                       [vbm_sv[:, 2]], **vbm_style, label='VBM')
            ax.scatter([cbm_sv[:, 0]], [cbm_sv[:, 1]],
                       [cbm_sv[:, 2]], **cbm_style, label='CBM')

        plt.legend(loc='upper center', ncol=3, prop={'size': 9})

    plt.tight_layout()
    plt.savefig(fname)


def GsWebpanel(result, row, key_descriptions):
    # for defect systems we don't want to show this panel
    if row.get('defect_name') is not None:
        return []

    parameter_description = _get_parameter_description(row)

    explained_keys = []

    def make_gap_row(name):
        value = result[name]
        description = _explain_bandgap(row, name)
        return [description, f'{value:0.2f} eV']

    gap_row = make_gap_row('gap')
    direct_gap_row = make_gap_row('gap_dir')

    for key in ['dipz', 'evacdiff', 'workfunction', 'dos_at_ef_soc']:
        if key in result.key_descriptions:
            key_description = result.key_descriptions[key]
            explanation = (f'{key_description} '
                           '(Including spin–orbit effects).\n\n'
                           + parameter_description)
            explained_key = describe_entry(key, description=explanation)
        else:
            explained_key = key
        explained_keys.append(explained_key)

    t = table(result, 'Property',
              explained_keys,
              key_descriptions)

    t['rows'] += [gap_row, direct_gap_row]

    if result.gap > 0:
        if result.get('evac'):
            eref = result.evac
            vbm_title = 'Valence band maximum relative to vacuum level'
            cbm_title = 'Conduction band minimum relative to vacuum level'
            reference_explanation = (
                'the asymptotic value of the '
                'electrostatic potential in the vacuum region')
        else:
            eref = result.efermi
            vbm_title = 'Valence band maximum relative to Fermi level'
            cbm_title = 'Conduction band minimum relative to Fermi level'
            reference_explanation = 'the Fermi level'

        vbm_displayvalue = result.vbm - eref
        cbm_displayvalue = result.cbm - eref
        info = [
            vbm_or_cbm_row(vbm_title, 'valence band maximum (VBM)',
                           reference_explanation, vbm_displayvalue),
            vbm_or_cbm_row(cbm_title, 'conduction band minimum (CBM)',
                           reference_explanation, cbm_displayvalue)
        ]

        t['rows'].extend(info)

    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)
    title = f'Basic electronic properties ({xcname})'

    panel = WebPanel(
        title=describe_entry(title, panel_description),
        columns=[[t], [fig('bz-with-gaps.png')]],
        sort=10)

    summary = WebPanel(
        title=describe_entry(
            'Summary',
            description='This panel contains a summary of '
            'basic properties of the material.'),
        columns=[[{
            'type': 'table',
            'header': ['Basic properties', ''],
            'rows': [gap_row],
        }]],
        plot_descriptions=[{'function': bz_with_band_extremums,
                            'filenames': ['bz-with-gaps.png']}],
        sort=10)

    return [panel, summary]


@prepare_result
class GapsResult(ASRResult):

    gap: float
    vbm: float
    cbm: float
    gap_dir: float
    vbm_dir: float
    cbm_dir: float
    k_vbm_c: typing.Tuple[float, float, float]
    k_cbm_c: typing.Tuple[float, float, float]
    k_vbm_dir_c: typing.Tuple[float, float, float]
    k_cbm_dir_c: typing.Tuple[float, float, float]
    skn1: typing.Tuple[int, int, int]
    skn2: typing.Tuple[int, int, int]
    skn1_dir: typing.Tuple[int, int, int]
    skn2_dir: typing.Tuple[int, int, int]
    efermi: float

    key_descriptions: typing.Dict[str, str] = dict(
        efermi='Fermi level [eV].',
        gap='Band gap [eV].',
        vbm='Valence band maximum [eV].',
        cbm='Conduction band minimum [eV].',
        gap_dir='Direct band gap [eV].',
        vbm_dir='Direct valence band maximum [eV].',
        cbm_dir='Direct conduction band minimum [eV].',
        k_vbm_c='Scaled k-point coordinates of valence band maximum (VBM).',
        k_cbm_c='Scaled k-point coordinates of conduction band minimum (CBM).',
        k_vbm_dir_c='Scaled k-point coordinates of direct valence band maximum (VBM).',
        k_cbm_dir_c='Scaled k-point coordinates of direct calence band minimum (CBM).',
        skn1="(spin,k-index,band-index)-tuple for valence band maximum.",
        skn2="(spin,k-index,band-index)-tuple for conduction band minimum.",
        skn1_dir="(spin,k-index,band-index)-tuple for direct valence band maximum.",
        skn2_dir="(spin,k-index,band-index)-tuple for direct conduction band minimum.",
    )


@prepare_result
class VacuumLevelResults(ASRResult):
    z_z: np.ndarray
    v_z: np.ndarray
    evacdiff: float
    dipz: float
    evac1: float
    evac2: float
    evacmean: float
    efermi_nosoc: float

    key_descriptions = {
        'z_z': 'Grid points for potential [Å].',
        'v_z': 'Electrostatic potential [eV].',
        'evacdiff': 'Difference of vacuum levels on both sides of slab [eV].',
        'dipz': 'Out-of-plane dipole [e · Å].',
        'evac1': 'Top side vacuum level [eV].',
        'evac2': 'Bottom side vacuum level [eV]',
        'evacmean': 'Average vacuum level [eV].',
        'efermi_nosoc': 'Fermi level without SOC [eV].'}


@prepare_result
class GsResult(ASRResult):
    """Container for ground state results.

    Examples
    --------
    >>> res = Result(data=dict(etot=0), strict=False)
    >>> res.etot
    0
    """

    forces: np.ndarray
    stresses: np.ndarray
    etot: float
    evac: float
    evacdiff: float
    dipz: float
    efermi: float
    gap: float
    vbm: float
    cbm: float
    gap_dir: float
    vbm_dir: float
    cbm_dir: float
    gap_dir_nosoc: float
    gap_nosoc: float
    gaps_nosoc: GapsResult
    k_vbm_c: typing.Tuple[float, float, float]
    k_cbm_c: typing.Tuple[float, float, float]
    k_vbm_dir_c: typing.Tuple[float, float, float]
    k_cbm_dir_c: typing.Tuple[float, float, float]
    skn1: typing.Tuple[int, int, int]
    skn2: typing.Tuple[int, int, int]
    skn1_dir: typing.Tuple[int, int, int]
    skn2_dir: typing.Tuple[int, int, int]
    workfunction: float
    vacuumlevels: VacuumLevelResults

    key_descriptions = dict(
        etot='Total energy [eV].',
        workfunction="Workfunction [eV]",
        forces='Forces on atoms [eV/Å].',
        stresses='Stress on unit cell [eV/Å^dim].',
        evac='Vacuum level [eV].',
        evacdiff='Vacuum level shift (Vacuum level shift) [eV].',
        dipz='Out-of-plane dipole [e · Å].',
        efermi='Fermi level [eV].',
        gap='Band gap [eV].',
        vbm='Valence band maximum [eV].',
        cbm='Conduction band minimum [eV].',
        gap_dir='Direct band gap [eV].',
        vbm_dir='Direct valence band maximum [eV].',
        cbm_dir='Direct conduction band minimum [eV].',
        gap_dir_nosoc='Direct gap without SOC [eV].',
        gap_nosoc='Gap without SOC [eV].',
        gaps_nosoc='Container for bandgap results without SOC.',
        vacuumlevels='Container for results that relate to vacuum levels.',
        k_vbm_c='Scaled k-point coordinates of valence band maximum (VBM).',
        k_cbm_c='Scaled k-point coordinates of conduction band minimum (CBM).',
        k_vbm_dir_c='Scaled k-point coordinates of direct valence band maximum (VBM).',
        k_cbm_dir_c='Scaled k-point coordinates of direct calence band minimum (CBM).',
        skn1="(spin,k-index,band-index)-tuple for valence band maximum.",
        skn2="(spin,k-index,band-index)-tuple for conduction band minimum.",
        skn1_dir="(spin,k-index,band-index)-tuple for direct valence band maximum.",
        skn2_dir="(spin,k-index,band-index)-tuple for direct conduction band minimum.",
    )

    formats = {"ase_webpanel": GsWebpanel}


######### gw #########
from asr.utils.gw_hse import GWHSEInfo
class GWInfo(GWHSEInfo):
    method_name = 'G₀W₀'
    name = 'gw'
    bs_filename = 'gw-bs.png'

    panel_description = make_panel_description(
        """The quasiparticle (QP) band structure calculated within the G₀W₀
approximation from a GGA starting point.
The treatment of frequency dependence is numerically exact. For
low-dimensional materials, a truncated Coulomb interaction is used to decouple
periodic images. The QP energies are extrapolated as 1/N to the infinite plane
wave basis set limit. Spin–orbit interactions are included
in post-process.""",
        articles=[
            'C2DB',
            href(
                """F. Rasmussen et al. Efficient many-body calculations for
two-dimensional materials using exact limits for the screened potential: Band gaps
of MoS2, h-BN, and phosphorene, Phys. Rev. B 94, 155406 (2016)""",
                'https://doi.org/10.1103/PhysRevB.94.155406',
            ),
            href(
                """A. Rasmussen et al. Towards fully automatized GW band structure
calculations: What we can learn from 60.000 self-energy evaluations,
arXiv:2009.00314""",
                'https://arxiv.org/abs/2009.00314v1'
            ),
        ]
    )

    band_gap_adjectives = 'quasi-particle'
    summary_sort = 12

    @staticmethod
    def plot_bs(row, filename):
        from asr.paneldata import plot_bs
        data = row.data['results-asr.gw.json']
        return plot_bs(row, filename=filename, bs_label='G₀W₀',
                       data=data,
                       efermi=data['efermi_gw_soc'],
                       cbm=row.get('cbm_gw'),
                       vbm=row.get('vbm_gw'))


def GwWebpanel(result, row, key_descriptions):
    from asr.utils.gw_hse import gw_hse_webpanel
    return gw_hse_webpanel(result, row, key_descriptions, GWInfo(row),
                           sort=16)


@prepare_result
class GwResult(ASRResult):
    from ase.spectrum.band_structure import BandStructure
    vbm_gw_nosoc: float
    cbm_gw_nosoc: float
    gap_dir_gw_nosoc: float
    gap_gw_nosoc: float
    kvbm_nosoc: typing.List[float]
    kcbm_nosoc: typing.List[float]
    vbm_gw: float
    cbm_gw: float
    gap_dir_gw: float
    gap_gw: float
    kvbm: typing.List[float]
    kcbm: typing.List[float]
    efermi_gw_nosoc: float
    efermi_gw_soc: float
    bandstructure: BandStructure
    key_descriptions = {
        "vbm_gw_nosoc": "Valence band maximum w/o soc. (G₀W₀) [eV]",
        "cbm_gw_nosoc": "Conduction band minimum w/o soc. (G₀W₀) [eV]",
        "gap_dir_gw_nosoc": "Direct gap w/o soc. (G₀W₀) [eV]",
        "gap_gw_nosoc": "Gap w/o soc. (G₀W₀) [eV]",
        "kvbm_nosoc": "k-point of G₀W₀ valence band maximum w/o soc",
        "kcbm_nosoc": "k-point of G₀W₀ conduction band minimum w/o soc",
        "vbm_gw": "Valence band maximum (G₀W₀) [eV]",
        "cbm_gw": "Conduction band minimum (G₀W₀) [eV]",
        "gap_dir_gw": "Direct band gap (G₀W₀) [eV]",
        "gap_gw": "Band gap (G₀W₀) [eV]",
        "kvbm": "k-point of G₀W₀ valence band maximum",
        "kcbm": "k-point of G₀W₀ conduction band minimum",
        "efermi_gw_nosoc": "Fermi level w/o soc. (G₀W₀) [eV]",
        "efermi_gw_soc": "Fermi level (G₀W₀) [eV]",
        "bandstructure": "GW bandstructure."
    }
    formats = {"ase_webpanel": GwWebpanel}


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


######### hyperfine #########
from ase.geometry import get_distances
panel_description = make_panel_description(
    """
Analysis of hyperfine coupling and spin coherence time.
""",
    articles=[
        href("""G. D. Cheng et al. Optical and spin coherence properties of NV
 center in diamond and 3C-SiC, Comp. Mat. Sc. 154, 60 (2018)""",
             'https://doi.org/10.1016/j.commatsci.2018.07.039'),
    ],
)


def get_atoms_close_to_center(center, atoms):
    """
    Return ordered list of the atoms closest to the defect.

    Note, that this is the case only if a previous defect calculation is present.
    Return list of atoms closest to the origin otherwise.
    """
    _, distances = get_distances(center, atoms.positions, cell=atoms.cell,
                                 pbc=atoms.pbc)
    args = np.argsort(distances[0])

    return args, distances[0][args]


def get_hf_table(hf_results, ordered_args):
    hf_array = np.zeros((10, 4))
    hf_atoms = []
    for i, arg in enumerate(ordered_args[:10]):
        hf_result = hf_results[arg]
        hf_atoms.append(hf_result['kind'] + ' (' + str(hf_result['index']) + ')')
        for j, value in enumerate([hf_result['magmom'], *hf_result['eigenvalues']]):
            hf_array[i, j] = f"{value:.2f}"

    rows = []
    for i, hf_tuple in enumerate(hf_array):
        rows.append((hf_atoms[i],
                     f'{hf_array[i][0]:.2f}',
                     f'{hf_array[i][1]:.2f} MHz',
                     f'{hf_array[i][2]:.2f} MHz',
                     f'{hf_array[i][3]:.2f} MHz',
                     ))

    table = {'type': 'table',
             'header': ['Nucleus (index)',
                        'Magn. moment',
                        'A<sub>1</sub>',
                        'A<sub>2</sub>',
                        'A<sub>3</sub>',
                        ]}

    table['rows'] = rows

    return table


gyromagnetic_ratios = {
    'H': (1, 42.577478),
    'He': (3, -32.434),
    'Li': (7, 16.546),
    'Be': (9, -6.298211),
    'B': (11, 13.6611),
    'C': (13, 10.7084),
    'N': (14, 3.077),
    'O': (17, -5.772),
    'F': (19, 40.052),
    'Ne': (21, -3.36275),
    'Na': (23, 11.262),
    'Mg': (25, -2.6084),
    'Al': (27, 11.103),
    'Si': (29, -8.465),
    'P': (31, 17.235),
    'S': (33, 3.27045),
    'Cl': (35, 4.17631),
    'K': (39, 1.98900),
    'Ca': (43, -2.86861),
    'Sc': (45, 10.35739),
    'Ti': (47, -2.40390),
    'V' : (51, 11.21232),
    'Cr': (53, -2.406290),
    'Mn': (55, 10.5163),
    'Fe': (57, 1.382),
    'Co': (59, 10.0532),
    'Ni': (61, -3.809960),
    'Cu': (63, 11.2952439),
    'Zn': (67, 2.668563),
    'Ga': (69, 10.23676),
    'Ge': (73, -1.48913),
    'As': (75, 7.312768),
    'Se': (77, 8.14828655),
    'Br': (79, 10.69908),
    'Kr': (83, -1.64398047),
    'Rb': (85, 4.1233194),
    'Sr': (89, -1.850870),
    'Y': (89, -2.0935685),
    'Zr': (91, -3.97213054),
    'Nb': (93, 10.44635),
    'Mo': (95, 2.7850588),
    'Ru': (101, -2.20099224),
    'Rh': (103, -1.34637703),
    'Ag': (107, -1.7299194),
    'Cd': (111, -9.0595),
    'In': (115, 9.3749856),
    'Sn': (119, -15.9365),
    'Sb': (121, 10.2418),
    'Te': (125, -13.5242),
    'I' : (127, 8.56477221),
    'Xe': (129, -11.8420),
    'Cs': (133, 5.614201),
    'Ba': (137, 4.755289),
    'Hf': (179, -1.08060),
    'Ta': (181, 5.1245083),
    'W': (183, 1.78243),
    'Re': (187, 9.76839),
    'Os': (189, 1.348764),
    'Ir': (193, 0.804325),
    'Pt': (195, 9.17955),
    'Au': (197, 0.73605),
    'Hg': (199, 7.66352),
    'Tl': (205, 24.8093),
    'Pb': (207, 8.8167),
    'Bi': (209, 6.91012),
    'La': (139, 6.049147)}
def get_gyro_table(row, result):
    """Return table with gyromagnetic ratios for each chemical element."""
    gyro_table = {'type': 'table',
                  'header': ['Nucleus', 'Isotope', 'Gyromagnetic ratio']}

    rows = []
    for i, g in enumerate(result.gfactors):
        rows.append((g['symbol'], gyromagnetic_ratios[g['symbol']][0],
                     f"{g['g']:.2f}"))
    gyro_table['rows'] = rows

    return gyro_table


def HFWebpanel(result, row, key_description):
    from asr.database.browser import (WebPanel,
                                      describe_entry)

    hf_results = result.hyperfine
    center = result.center
    if center[0] is None:
        center = [0, 0, 0]

    atoms = row.toatoms()
    args, distances = get_atoms_close_to_center(center, atoms)

    hf_table = get_hf_table(hf_results, args)
    gyro_table = get_gyro_table(row, result)

    hyperfine = WebPanel(describe_entry('Hyperfine (HF) parameters',
                                        panel_description),
                         columns=[[hf_table], [gyro_table]],
                         sort=42)

    return [hyperfine]


@prepare_result
class HyperfineResult(ASRResult):
    """Container for hyperfine coupling results."""

    index: int
    kind: str
    magmom: float
    eigenvalues: typing.Tuple[float, float, float]

    key_descriptions: typing.Dict[str, str] = dict(
        index='Atom index.',
        kind='Atom type.',
        magmom='Magnetic moment.',
        eigenvalues='Tuple of the three main HF components [MHz].'
    )


@prepare_result
class GyromagneticResult(ASRResult):
    """Container for gyromagnetic factor results."""

    symbol: str
    g: float

    key_descriptions: typing.Dict[str, str] = dict(
        symbol='Atomic species.',
        g='g-factor for the isotope.'
    )

    @classmethod
    def fromdict(cls, dct):
        gyro_results = []
        for symbol, g in dct.items():
            gyro_result = GyromagneticResult.fromdata(
                symbol=symbol,
                g=g)
        gyro_results.append(gyro_result)

        return gyro_results


@prepare_result
class HFResult(ASRResult):
    """Container for asr.hyperfine results."""

    hyperfine: typing.List[HyperfineResult]
    gfactors: typing.List[GyromagneticResult]
    center: typing.Tuple[float, float, float]
    delta_E_hyp: float
    sc_time: float

    key_descriptions: typing.Dict[str, str] = dict(
        hyperfine='List of HyperfineResult objects for all atoms.',
        gfactors='List of GyromagneticResult objects for each atom species.',
        center='Center to show values on webpanel (only relevant for defects).',
        delta_E_hyp='Hyperfine interaction energy [eV].',
        sc_time='Spin coherence time [s].')

    formats = {'ase_webpanel': HFWebpanel}


######### magnetic_anisotropy #########
def equation():
    i = '<sub>i</sub>'
    j = '<sub>j</sub>'
    z = '<sup>z</sup>'
    return (f'E{i} = '
            f'−1/2 J ∑{j} S{i} S{j} '
            f'− 1/2 B ∑{j} S{i}{z} S{j}{z} '
            f'− A S{i}{z} S{i}{z}')
# We don't have mathjax I think, so we should probably use either html
# or unicode.  But z does not exist as unicode superscript, so we mostly
# use html for sub/superscripts.

# This panel description actually assumes that we also have results for the
# exchange recipe.


panel_description = make_panel_description(
    """
Heisenberg parameters, magnetic anisotropy and local magnetic
moments. The Heisenberg parameters were calculated assuming that the
magnetic energy of atom i can be represented as

  {equation},

where J is the exchange coupling, B is anisotropic exchange, A is
single-ion anisotropy and the sums run over nearest neighbours. The
magnetic anisotropy was obtained from non-selfconsistent spin-orbit
calculations where the exchange-correlation magnetic field from a
scalar calculation was aligned with the x, y and z directions.

""".format(equation=equation()),
    articles=[
        'C2DB',
        href("""D. Torelli et al. High throughput computational screening for 2D
ferromagnetic materials: the critical role of anisotropy and local
correlations, 2D Mater. 6 045018 (2019)""",
             'https://doi.org/10.1088/2053-1583/ab2c43'),
    ],
)


def MagAniWebpanel(result, row, key_descriptions):
    if row.get('magstate', 'NM') == 'NM':
        return []

    magtable = table(row, 'Property',
                     ['magstate', 'magmom',
                      'dE_zx', 'dE_zy'], kd=key_descriptions)

    from asr.utils.hacks import gs_xcname_from_row
    xcname = gs_xcname_from_row(row)

    panel = {'title':
             describe_entry(
                 f'Basic magnetic properties ({xcname})',
                 panel_description),
             'columns': [[magtable], []],
             'sort': 11}
    return [panel]


@prepare_result
class MagAniResult(ASRResult):

    spin_axis: str
    E_x: float
    E_y: float
    E_z: float
    theta: float
    phi: float
    dE_zx: float
    dE_zy: float

    key_descriptions = {
        "spin_axis": "Magnetic easy axis",
        "E_x": "Soc. total energy, x-direction [meV/unit cell]",
        "E_y": "Soc. total energy, y-direction [meV/unit cell]",
        "E_z": "Soc. total energy, z-direction [meV/unit cell]",
        "theta": "Easy axis, polar coordinates, theta [radians]",
        "phi": "Easy axis, polar coordinates, phi [radians]",
        "dE_zx":
        "Magnetic anisotropy energy between x and z axis [meV/unit cell]",
        "dE_zy":
        "Magnetic anisotropy energy between y and z axis [meV/unit cell]"
    }

    formats = {"ase_webpanel": MagAniWebpanel}


######### magstate #########
atomic_mom_threshold = 0.1
def MagStateWebpanel(result, row, key_descriptions):
    """Webpanel for magnetic state."""
    from asr.database.browser import describe_entry, dl, code, WebPanel

    is_magnetic = describe_entry(
        'Magnetic',
        'Is material magnetic?'
        + dl(
            [
                [
                    'Magnetic',
                    code('if max(abs(atomic_magnetic_moments)) > '
                         f'{atomic_mom_threshold}')
                ],
                [
                    'Not magnetic',
                    code('otherwise'),
                ],
            ]
        )
    )

    yesno = ['No', 'Yes'][row.is_magnetic]

    rows = [[is_magnetic, yesno]]
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': ['Basic properties', ''],
                             'rows': rows}]],
               'sort': 0}

    """
    It makes sense to write the local orbital magnetic moments in the same
    table as the previous local spin magnetic moments; however, orbmag.py was
    added much later than magstate.py, so in order to accomplish this without
    inconvenient changes that may affect other people's projects, we need to
    load the orbmag.py results in this 'hacky' way
    """
    results_orbmag = row.data.get('results-asr.orbmag.json')
    if result.magstate == 'NM':
        return [summary]
    else:
        magmoms_header = ['Atom index', 'Atom type',
                          'Local spin magnetic moment (μ<sub>B</sub>)',
                          'Local orbital magnetic moment (μ<sub>B</sub>)']
        if results_orbmag is None:
            magmoms_rows = [[str(a), symbol, f'{magmom:.3f}', '--']
                            for a, (symbol, magmom)
                            in enumerate(zip(row.get('symbols'),
                                             result.magmoms))]
        else:
            magmoms_rows = [[str(a), symbol, f'{magmom:.3f}', f'{orbmag:.3f}']
                            for a, (symbol, magmom, orbmag)
                            in enumerate(zip(row.get('symbols'),
                                             result.magmoms,
                                             results_orbmag['orbmag_a']))]

        magmoms_table = {'type': 'table',
                         'header': magmoms_header,
                         'rows': magmoms_rows}

        from asr.utils.hacks import gs_xcname_from_row
        xcname = gs_xcname_from_row(row)
        panel = WebPanel(title=f'Basic magnetic properties ({xcname})',
                         columns=[[], [magmoms_table]], sort=11)

        return [summary, panel]


@prepare_result
class MagStateResult(ASRResult):

    magstate: str
    is_magnetic: bool
    magmoms: List[float]
    magmom: float
    nspins: int

    key_descriptions = {'magstate': 'Magnetic state.',
                        'is_magnetic': 'Is the material magnetic?',
                        'magmoms': 'Atomic magnetic moments.',
                        'magmom': 'Total magnetic moment.',
                        'nspins': 'Number of spins in system.'}

    formats = {"ase_webpanel": MagStateWebpanel}


######### orbmag #########
@prepare_result
class OrbMagResult(ASRResult):

    orbmag_a: Optional[List[float]]
    orbmag_sum: Optional[float]
    orbmag_max: Optional[float]

    key_descriptions = {
        "orbmag_a": "Local orbital magnetic moments along easy axis [μ_B]",
        "orbmag_sum": "Sum of local orbital magnetic moments [μ_B]",
        "orbmag_max": "Maximum norm of local orbital magnetic moments [μ_B]"
    }


######### pdos #########
def PdosWebpanel(result, row, key_descriptions):
    from asr.database.browser import (fig,
                                      entry_parameter_description,
                                      describe_entry, WebPanel)
    from asr.utils.hacks import gs_xcname_from_row

    # PDOS figure
    parameter_description = entry_parameter_description(
        row.data,
        'asr.pdos@calculate')
    dependencies_parameter_descriptions = ''
    for dependency, exclude_keys in zip(
            ['asr.gs@calculate'],
            [set(['txt', 'fixdensity', 'verbose', 'symmetry',
                  'idiotproof', 'maxiter', 'hund', 'random',
                  'experimental', 'basis', 'setups'])]
    ):
        epd = entry_parameter_description(
            row.data,
            dependency,
            exclude_keys=exclude_keys)
        dependencies_parameter_descriptions += f'\n{epd}'
    explanation = ('Orbital projected density of states without spin–orbit coupling\n\n'
                   + parameter_description
                   + dependencies_parameter_descriptions)

    xcname = gs_xcname_from_row(row)
    # Projected band structure and DOS panel
    panel = WebPanel(
        title=f'Projected band structure and DOS ({xcname})',
        columns=[[],
                 [describe_entry(fig(pdos_figfile, link='empty'),
                                 description=explanation)]],
        plot_descriptions=[{'function': plot_pdos_nosoc,
                            'filenames': [pdos_figfile]}],
        sort=13)

    return [panel]


pdos_figfile = 'scf-pdos_nosoc.png'


# ----- Fast steps ----- #
@prepare_result
class PdosResult(ASRResult):

    efermi: float
    symbols: typing.List[str]
    energies: typing.List[float]
    pdos_syl: typing.List[float]

    key_descriptions: typing.Dict[str, str] = dict(
        efermi="Fermi level [eV] of ground state with dense k-mesh.",
        symbols="Chemical symbols.",
        energies="Energy mesh of pdos results.",
        pdos_syl=("Projected density of states [states / eV] for every set of keys "
                  "'s,y,l', that is spin, symbol and orbital l-quantum number.")
    )


@prepare_result
class PDResult(ASRResult):

    dos_at_ef_nosoc: float
    dos_at_ef_soc: float
    pdos_nosoc: PdosResult
    pdos_soc: PdosResult

    key_descriptions: typing.Dict[str, str] = dict(
        dos_at_ef_nosoc=("Density of states at the Fermi "
                         "level w/o soc [states / (unit cell * eV)]"),
        dos_at_ef_soc=("Density of states at the Fermi "
                       "level [states / (unit cell * eV)])"),
        pdos_nosoc="Projected density of states w/o soc.",
        pdos_soc="Projected density of states"
    )
    formats = {"ase_webpanel": PdosWebpanel}

# ---------- Plotting ---------- #
from collections import defaultdict


def get_ordered_syl_dict(dct_syl, symbols):
    """Order a dictionary with syl keys.

    Parameters
    ----------
    dct_syl : dict
        Dictionary with keys f'{s},{y},{l}'
        (spin (s), chemical symbol (y), angular momentum (l))
    symbols : list
        Sort symbols after index in this list

    Returns
    -------
    outdct_syl : OrderedDict
        Sorted dct_syl

    """
    from collections import OrderedDict

    # Setup ssili (spin, symbol index, angular momentum index) key
    def ssili(syl):
        s, y, L = syl.split(',')
        # Symbols list can have multiple entries of the same symbol
        # ex. ['O', 'Fe', 'O']. In this case 'O' will have index 0 and
        # 'Fe' will have index 1.
        si = symbols.index(y)
        li = ['s', 'p', 'd', 'f'].index(L)
        return f'{s}{si}{li}'

    return OrderedDict(sorted(dct_syl.items(), key=lambda t: ssili(t[0])))


def get_yl_colors(dct_syl):
    """Get the color indices corresponding to each symbol and angular momentum.

    Parameters
    ----------
    dct_syl : OrderedDict
        Ordered dictionary with keys f'{s},{y},{l}'
        (spin (s), chemical symbol (y), angular momentum (l))

    Returns
    -------
    color_yl : OrderedDict
        Color strings for each symbol and angular momentum

    """
    from collections import OrderedDict

    color_yl = OrderedDict()
    c = 0
    for key in dct_syl:
        # Do not differentiate spin by color
        if int(key[0]) == 0:  # if spin is 0
            color_yl[key[2:]] = 'C{}'.format(c)
            c += 1
            c = c % 10  # only 10 colors available in cycler

    return color_yl


def plot_pdos_nosoc(*args, **kwargs):
    return plot_pdos(*args, soc=False, **kwargs)


def plot_pdos_soc(*args, **kwargs):
    return plot_pdos(*args, soc=True, **kwargs)


def plot_pdos(row, filename, soc=True,
              figsize=(5.5, 5), lw=1):

    def smooth(y, npts=3):
        return np.convolve(y, np.ones(npts) / npts, mode='same')

    # Check if pdos data is stored in row
    results = 'results-asr.pdos.json'
    pdos = 'pdos_soc' if soc else 'pdos_nosoc'
    if results in row.data and pdos in row.data[results]:
        data = row.data[results][pdos]
    else:
        return

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.patheffects as path_effects

    # Extract raw data
    symbols = data['symbols']
    pdos_syl = get_ordered_syl_dict(data['pdos_syl'], symbols)
    e_e = data['energies'].copy() - row.get('evac', 0)
    ef = data['efermi']

    # Find energy range to plot in
    if soc:
        emin = row.get('vbm', ef) - 3 - row.get('evac', 0)
        emax = row.get('cbm', ef) + 3 - row.get('evac', 0)
    else:
        nosoc_data = row.data['results-asr.gs.json']['gaps_nosoc']
        vbmnosoc = nosoc_data.get('vbm', ef)
        cbmnosoc = nosoc_data.get('cbm', ef)

        if vbmnosoc is None:
            vbmnosoc = ef

        if cbmnosoc is None:
            cbmnosoc = ef

        emin = vbmnosoc - 3 - row.get('evac', 0)
        emax = cbmnosoc + 3 - row.get('evac', 0)

    # Set up energy range to plot in
    i1, i2 = abs(e_e - emin).argmin(), abs(e_e - emax).argmin()

    # Get color code
    color_yl = get_yl_colors(pdos_syl)

    # Figure out if pdos has been calculated for more than one spin channel
    spinpol = False
    for k in pdos_syl.keys():
        if int(k[0]) == 1:
            spinpol = True
            break

    # Set up plot
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot pdos
    pdosint_s = defaultdict(float)
    for key in pdos_syl:
        pdos = pdos_syl[key]
        spin, symbol, lstr = key.split(',')
        spin = int(spin)
        sign = 1 if spin == 0 else -1

        # Integrate pdos to find suiting pdos range
        pdosint_s[spin] += np.trapz(y=pdos[i1:i2], x=e_e[i1:i2])

        # Label atomic symbol and angular momentum
        if spin == 0:
            label = '{} ({})'.format(symbol, lstr)
        else:
            label = None

        ax.plot(smooth(pdos) * sign, e_e,
                label=label, color=color_yl[key[2:]])

    ax.axhline(ef - row.get('evac', 0), color='k', ls=':')

    # Set up axis limits
    ax.set_ylim(emin, emax)
    if spinpol:  # Use symmetric limits
        xmax = max(pdosint_s.values())
        ax.set_xlim(-xmax * 0.5, xmax * 0.5)
    else:
        ax.set_xlim(0, pdosint_s[0] * 0.5)

    # Annotate E_F
    xlim = ax.get_xlim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.99
    text = plt.text(x0, ef - row.get('evac', 0),
                    r'$E_\mathrm{F}$',
                    fontsize=rcParams['font.size'] * 1.25,
                    ha='right',
                    va='bottom')

    text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    ax.set_xlabel('Projected DOS [states / eV]')
    if row.get('evac') is not None:
        ax.set_ylabel(r'$E-E_\mathrm{vac}$ [eV]')
    else:
        ax.set_ylabel(r'$E$ [eV]')

    # Set up legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()


######### phonons #########
panel_description = make_panel_description(
    """
The Gamma-point phonons of a supercell containing the primitive unit cell
repeated 2 times along each periodic direction. In the Brillouin zone (BZ) of
the primitive cell, this yields the phonons at the Gamma-point and
high-symmetry points at the BZ boundary. A negative eigenvalue of the Hessian
matrix (the second derivative of the energy w.r.t. to atomic displacements)
indicates a dynamical instability.
""",
    articles=['C2DB'],
)


def plot_bandstructure(row, fname):
    from matplotlib import pyplot as plt
    from ase.spectrum.band_structure import BandStructure
    data = row.data.get('results-asr.phonons.json')
    path = data['path']
    energies = data['interp_freqs_kl'] * 1e3
    exact_indices = []
    for q_c in data['q_qc']:
        diff_kc = path.kpts - q_c
        diff_kc -= np.round(diff_kc)
        inds = np.argwhere(np.all(np.abs(diff_kc) < 1e-3, 1))
        exact_indices.extend(inds.tolist())

    en_exact = np.zeros_like(energies) + np.nan
    for ind in exact_indices:
        en_exact[ind] = energies[ind]

    bs = BandStructure(path=path, energies=en_exact[None])
    bs.plot(ax=plt.gca(), ls='', marker='o', colors=['C1'],
            emin=np.min(energies * 1.1), emax=np.max([np.max(energies * 1.15),
                                                      0.0001]),
            ylabel='Phonon frequencies [meV]')
    plt.tight_layout()
    plt.savefig(fname)


def PhononWebpanel(result, row, key_descriptions):
    phonontable = table(row, 'Property', ['minhessianeig'], key_descriptions)

    panel = {'title': describe_entry('Phonons', panel_description),
             'columns': [[fig('phonon_bs.png')], [phonontable]],
             'plot_descriptions': [{'function': plot_bandstructure,
                                    'filenames': ['phonon_bs.png']}],
             'sort': 3}

    return [panel]


@prepare_result
class PhononResult(ASRResult):
    minhessianeig: float
    dynamic_stability_phonons: str
    q_qc: typing.List[typing.Tuple[float, float, float]]
    omega_kl: typing.List[typing.List[float]]
    path: BandPath
    modes_kl: typing.List[typing.List[float]]
    interp_freqs_kl: typing.List[typing.List[float]]

    key_descriptions = {
        "minhessianeig": "Minimum eigenvalue of Hessian [eV/Å²]",
        "dynamic_stability_phonons": "Phonon dynamic stability (low/high)",
        "q_qc": "List of momenta consistent with supercell.",
        "omega_kl": "Phonon frequencies.",
        "modes_kl": "Phonon modes.",
        "interp_freqs_kl": "Interpolated phonon frequencies.",
        "path": "Interpolated phonon bandstructure path.",
    }
    formats = {"ase_webpanel": PhononWebpanel}


######### phonopy #########
def PhonopyWebpanel(result, row, key_descriptions):
    from asr.database.browser import table, fig

    phonontable = table(row, "Property", ["minhessianeig"], key_descriptions)

    panel = {
        "title": "Phonon bandstructure",
        "columns": [[fig("phonon_bs.png")], [phonontable]],
        "plot_descriptions": [
            {"function": plot_bandstructure, "filenames": ["phonon_bs.png"]}
        ],
        "sort": 3,
    }

    dynstab = row.get("dynamic_stability_level")
    stabilities = {1: "low", 2: "medium", 3: "high"}
    high = "Minimum eigenvalue of Hessian > -0.01 meV/Å² AND elastic const. > 0"
    medium = "Minimum eigenvalue of Hessian > -2 eV/Å² AND elastic const. > 0"
    low = "Minimum eigenvalue of Hessian < -2 eV/Å² OR elastic const. < 0"
    row = [
        "Phonons",
        '<a href="#" data-toggle="tooltip" data-html="true" '
        + 'title="LOW: {}&#13;MEDIUM: {}&#13;HIGH: {}">{}</a>'.format(
            low, medium, high, stabilities[dynstab].upper()
        ),
    ]

    summary = {
        "title": "Summary",
        "columns": [
            [
                {
                    "type": "table",
                    "header": ["Stability", "Category"],
                    "rows": [row],
                }
            ]
        ],
    }
    return [panel, summary]


@prepare_result
class PhonopyResult(ASRResult):
    omega_kl: typing.List[typing.List[float]]
    minhessianeig: float
    eigs_kl: typing.List[typing.List[complex]]
    q_qc: typing.List[typing.Tuple[float, float, float]]
    phi_anv: typing.List[typing.List[typing.List[float]]]
    u_klav: typing.List[typing.List[float]]
    irr_l: typing.List[str]
    path: BandPath
    dynamic_stability_level: int

    key_descriptions = {
        "omega_kl": "Phonon frequencies.",
        "minhessianeig": "Minimum eigenvalue of Hessian [eV/Å²]",
        "eigs_kl": "Dynamical matrix eigenvalues.",
        "q_qc": "List of momenta consistent with supercell.",
        "phi_anv": "Force constants.",
        "u_klav": "Phonon modes.",
        "irr_l": "Phonon irreducible representations.",
        "path": "Phonon bandstructure path.",
        "dynamic_stability_level": "Phonon dynamic stability (1,2,3)",
    }

    formats = {"ase_webpanel": PhonopyWebpanel}


######### piezoelectrictensor #########
panel_description = make_panel_description("""
The piezoelectric tensor, c, is a rank-3 tensor relating the macroscopic
polarization to an applied strain. In Voigt notation, c is expressed as a 3xN
matrix relating the (x,y,z) components of the polarizability to the N
independent components of the strain tensor. The polarization in a periodic
direction is calculated as an integral over Berry phases. The polarization in a
non-periodic direction is obtained by direct evaluation of the first moment of
the electron density.
""")

all_voigt_labels = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
all_voigt_indices = [[0, 1, 2, 1, 0, 0],
                     [0, 1, 2, 2, 2, 1]]

def get_voigt_mask(pbc_c: typing.List[bool]):
    non_pbc_axes = set(char for char, pbc in zip('xyz', pbc_c) if not pbc)

    mask = [False
            if set(voigt_label).intersection(non_pbc_axes)
            else True
            for voigt_label in all_voigt_labels]
    return mask

def get_voigt_indices(pbc: typing.List[bool]):
    mask = get_voigt_mask(pbc)
    return [list(itertools.compress(indices, mask)) for indices in all_voigt_indices]


def get_voigt_labels(pbc: typing.List[bool]):
    mask = get_voigt_mask(pbc)
    return list(itertools.compress(all_voigt_labels, mask))



def PiezoEleTenWebpanel(result, row, key_descriptions):

    piezodata = row.data['results-asr.piezoelectrictensor.json']
    e_vvv = piezodata['eps_vvv']
    e0_vvv = piezodata['eps_clamped_vvv']

    voigt_indices = get_voigt_indices(row.pbc)
    voigt_labels = get_voigt_labels(row.pbc)

    e_ij = e_vvv[:,
                 voigt_indices[0],
                 voigt_indices[1]]
    e0_ij = e0_vvv[:,
                   voigt_indices[0],
                   voigt_indices[1]]

    etable = matrixtable(e_ij,
                         columnlabels=voigt_labels,
                         rowlabels=['x', 'y', 'z'],
                         title='c<sub>ij</sub> (e/Å<sup>dim-1</sup>)')

    e0table = matrixtable(
        e0_ij,
        columnlabels=voigt_labels,
        rowlabels=['x', 'y', 'z'],
        title='c<sup>clamped</sup><sub>ij</sub> (e/Å<sup>dim-1</sup>)')

    columns = [[etable], [e0table]]

    panel = {'title': describe_entry('Piezoelectric tensor',
                                     panel_description),
             'columns': columns}

    return [panel]


@prepare_result
class PiezoEleTenResult(ASRResult):

    eps_vvv: typing.List[typing.List[typing.List[float]]]
    eps_clamped_vvv: typing.List[typing.List[typing.List[float]]]

    key_descriptions = {'eps_vvv': 'Piezoelectric tensor.',
                        'eps_clamped_vvv': 'Piezoelectric tensor.'}
    formats = {"ase_webpanel": PiezoEleTenWebpanel}


######### projected_bandstructure #########
panel_description = make_panel_description(
    """The single-particle band structure and density of states projected onto
atomic orbitals (s,p,d). Spin–orbit interactions are not included in these
plots.""",
    articles=[
        'C2DB',
    ],
)


scf_projected_bs_filename = 'scf-projected-bs.png'


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


def projected_bs_scf(row, filename,
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
    import numpy as np
    from ase.spectrum.band_structure import BandStructure, BandStructurePlot

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

    # Take bands with energies in range
    e_skn = d['bs_nosoc']['energies']
    inrange_skn = np.logical_and(e_skn > emin, e_skn < emax)
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
             ylabel=label, **style)

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


def ProjBSWebpanel(result, row, key_descriptions):
    xcname = gs_xcname_from_row(row)

    # Projected band structure figure
    parameter_description = entry_parameter_description(
        row.data,
        'asr.bandstructure@calculate')
    dependencies_parameter_descriptions = ''
    for dependency, exclude_keys in zip(
            ['asr.gs@calculate'],
            [set(['txt', 'fixdensity', 'verbose', 'symmetry',
                  'idiotproof', 'maxiter', 'hund', 'random',
                  'experimental', 'basis', 'setups'])]
    ):
        epd = entry_parameter_description(
            row.data,
            dependency,
            exclude_keys=exclude_keys)
        dependencies_parameter_descriptions += f'\n{epd}'
    explanation = ('Orbital projected band structure without spin–orbit coupling\n\n'
                   + parameter_description
                   + dependencies_parameter_descriptions)

    panel = WebPanel(
        title=describe_entry(
            f'Projected band structure and DOS ({xcname})',
            panel_description),
        columns=[[describe_entry(fig(scf_projected_bs_filename, link='empty'),
                                 description=explanation)],
                 [fig('bz-with-gaps.png')]],
        plot_descriptions=[{'function': projected_bs_scf,
                            'filenames': [scf_projected_bs_filename]}],
        sort=13.5)

    return [panel]


@prepare_result
class ProjBSResult(ASRResult):

    symbols: typing.List[str]
    yl_i: typing.List[typing.Tuple[str, str]]
    weight_skni: typing.List[typing.List[typing.List[float]]]

    key_descriptions: typing.Dict[str, str] = dict(
        symbols="Chemical symbols.",
        yl_i="Symbol and orbital angular momentum string ('y,l') of each orbital i.",
        weight_skni="Weight of each projector (indexed by (s, k, n)) on orbitals i.",
    )

    formats = {'ase_webpanel': ProjBSWebpanel}


######### raman #########
panel_description = make_panel_description(
    """Raman spectroscopy relies on inelastic scattering of photons by optical
phonons. The Stokes part of the Raman spectrum, corresponding to emission of a
single Gamma-point phonon is calculated for different incoming/outgoing photon
polarizations using third order perturbation theory.""",
    articles=[
        href("""A. Taghizadeh et al.  A library of ab initio Raman spectra for automated
identification of 2D materials. Nat Commun 11, 3011 (2020).""",
             'https://doi.org/10.1038/s41467-020-16529-6'),
    ],
)

# Count the modes and their degeneracy factors


def count_deg(freqs_l, freq_err=2):

    # Degeneracy factor for modes
    w_l = [freqs_l[0]]
    rep_l = [1]
    # Loop over modes
    for wss in freqs_l[1:]:
        ind = len(w_l) - 1
        if np.abs(w_l[ind] - wss) > freq_err:
            w_l.append(wss)
            rep_l.append(1)
        else:
            rep_l[ind] += 1
    w_l = np.array(w_l)
    rep_l = np.array(rep_l)
    # Return the output
    return w_l, rep_l


def raman(row, filename):
    # Import the required modules
    import matplotlib.pyplot as plt

    # All required settings
    params = {'broadening': 3.0,  # in cm^-1
              'wavelength': 532.0,  # in nm
              'polarization': ['xx', 'yy', 'zz'],
              'temperature': 300}

    # Read the data from the disk
    data = row.data.get('results-asr.raman.json')

    # If no data, return
    if data is None:
        return

    # Lorentzian function definition
    def lor(w, g):
        lor = 0.5 * g / (np.pi * ((w.real)**2 + 0.25 * g**2))
        return lor
    from math import pi, sqrt
    # Gaussian function definition

    def gauss(w, g):
        gauss = 1 / (g * sqrt(2 * pi)) * np.exp(-0.5 * w**2 / g**2)
        gauss[gauss < 1e-16] = 0
        return gauss

    # Compute spectrum based on a set of resonances
    from ase.units import kB
    cm = 1 / 8065.544
    kbT = kB * params['temperature'] / cm

    def calcspectrum(wlist, rlist, ww, gamma=10, shift=0, kbT=kbT):
        rr = np.zeros(np.size(ww))
        for wi, ri in zip(wlist, rlist):
            if wi > 1e-1:
                nw = 1 / (np.exp(wi / kbT) - 1)
                curr = (1 + nw) * np.abs(ri)**2
                rr = rr + curr * gauss(ww - wi - shift, gamma)
        return rr

    # Make a latex type formula
    def getformula(matstr):
        matformula = r''
        for ch in matstr:
            if ch.isdigit():
                matformula += '$_' + ch + '$'
            else:
                matformula += ch
        return matformula

    # Set the variables and parameters
    wavelength_w = data['wavelength_w']
    freqs_l = data['freqs_l']
    amplitudes_vvwl = data['amplitudes_vvwl']
    selpol = params['polarization']
    gamma = params['broadening']

    # If the wavelength was not found, return
    waveind = int(np.where(wavelength_w == params['wavelength'])[0])
    if not waveind:
        return

    # Check the data to be consistent
    ampshape = amplitudes_vvwl.shape
    freqshape = len(freqs_l)
    waveshape = len(wavelength_w)
    if (ampshape[0] != 3) or (ampshape[1] != 3) or (
            ampshape[2] != waveshape) or (ampshape[3] != freqshape):
        return

    # Make the spectrum
    maxw = min([int(np.max(freqs_l) + 200), int(1.2 * np.max(freqs_l))])
    minw = -maxw / 100
    ww = np.linspace(minw, maxw, 2 * maxw)
    rr = {}
    maxr = np.zeros(len(selpol))
    for ii, pol in enumerate(selpol):
        d_i = 0 * (pol[0] == 'x') + 1 * (pol[0] == 'y') + 2 * (pol[0] == 'z')
        d_o = 0 * (pol[1] == 'x') + 1 * (pol[1] == 'y') + 2 * (pol[1] == 'z')
        rr[pol] = calcspectrum(
            freqs_l, amplitudes_vvwl[d_i, d_o, waveind], ww, gamma=gamma)
        maxr[ii] = np.max(rr[pol])

    # Make the figure panel and add y=0 axis
    ax = plt.figure().add_subplot(111)
    ax.axhline(y=0, color="k")

    # Plot the data and add the axis labels
    for ipol, pol in enumerate(selpol):
        ax.plot(ww, rr[pol] / np.max(maxr), c='C' + str(ipol), label=pol)
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Raman intensity (a.u.)')
    ax.set_ylim((-0.1, 1.1))
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim((minw, maxw))

    # Add the legend to figure
    ax.legend()

    # Count the modes and their degeneracy factors
    w_l, rep_l = count_deg(freqs_l)

    # Add the phonon bars to the figure with showing their degeneracy factors
    pltbar = plt.bar(w_l, -0.04, width=maxw / 100, color='k')
    for idx, rect in enumerate(pltbar):
        ax.text(rect.get_x() + rect.get_width() / 2., -0.1,
                str(int(rep_l[idx])), ha='center', va='bottom', rotation=0)

    # Remove the extra space and save the figure
    plt.tight_layout()
    plt.savefig(filename)


def RamanWebpanel(result, row, key_descriptions):

    # Make a table from the phonon modes
    data = row.data.get('results-asr.raman.json')
    if data:
        table = []
        freqs_l = data['freqs_l']
        w_l, rep_l = count_deg(freqs_l)
        # print(w_l)
        # print(rep_l)
        nph = len(w_l)
        for ii in range(nph):
            key = 'Mode {}'.format(ii + 1)
            table.append(
                (key,
                 np.array2string(
                     np.abs(
                         w_l[ii]),
                     precision=1),
                    rep_l[ii]))
        opt = {'type': 'table',
               'header': ['Mode', 'Frequency (1/cm)', 'Degeneracy'],
               'rows': table}
    else:
        opt = None
    # Make the panel
    panel = {'title': describe_entry('Raman spectrum', panel_description),
             'columns': [[fig('Raman.png')], [opt]],
             'plot_descriptions':
                 [{'function': raman,
                   'filenames': ['Raman.png']}],
             'sort': 22}

    return [panel]


@prepare_result
class RamanResult(ASRResult):

    freqs_l: typing.List[float]
    wavelength_w: typing.List[float]
    amplitudes_vvwl: typing.List[typing.List[typing.List[typing.List[complex]]]]

    key_descriptions = {
        "freqs_l": "Phonon frequencies (the Gamma point) [1/cm]",
        "wavelength_w": "Laser excitation wavelength [nm]",
        "amplitudes_vvwl": "Raman tensor [a.u.]",
    }
    formats = {"ase_webpanel": RamanWebpanel}


######### relax #########
from ase import Atoms
@prepare_result
class RelaxResult(ASRResult):
    """Result class for :py:func:`asr.relax.main`."""

    version: int = 0

    atoms: Atoms
    images: typing.List[Atoms]
    etot: float
    edft: float
    spos: np.ndarray
    symbols: typing.List[str]
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    key_descriptions = \
        {'atoms': 'Relaxed atomic structure.',
         'images': 'Path taken when relaxing structure.',
         'etot': 'Total energy [eV]',
         'edft': 'DFT total energy [eV]',
         'spos': 'Array: Scaled positions',
         'symbols': 'Array: Chemical symbols',
         'a': 'Cell parameter a [Å]',
         'b': 'Cell parameter b [Å]',
         'c': 'Cell parameter c [Å]',
         'alpha': 'Cell parameter alpha [deg]',
         'beta': 'Cell parameter beta [deg]',
         'gamma': 'Cell parameter gamma [deg]'}


######### SHG #########
class CentroSymmetric(Exception):
    """CentroSymmetric crystals have vanishing SHG response!."""

    pass

def make_full_chi(sym_chi, chi_dict):

    if len(sym_chi) == 1:
        return 0

    # Make the full chi from its symmetries
    for pol in sorted(sym_chi.keys()):
        if pol != 'zero':
            chidata = chi_dict[pol]
            nw = len(chidata)
    chi_vvvl = np.zeros((3, 3, 3, nw), complex)
    for pol in sorted(sym_chi.keys()):
        relation = sym_chi.get(pol)
        if pol == 'zero':
            if relation != '':
                for zpol in relation.split('='):
                    ind = ['xyz'.index(zpol[ii]) for ii in range(3)]
                    chi_vvvl[ind[0], ind[1], ind[2]] = np.zeros((nw), complex)
        else:
            chidata = chi_dict[pol]
            chidata = chidata[1]
            for zpol in relation.split('='):
                if zpol[0] == '-':
                    ind = ['xyz'.index(zpol[ii + 1]) for ii in range(3)]
                    chi_vvvl[ind[0], ind[1], ind[2]] = -chidata
                else:
                    ind = ['xyz'.index(zpol[ii]) for ii in range(3)]
                    chi_vvvl[ind[0], ind[1], ind[2]] = chidata

    return chi_vvvl


def plot_shg(row, *filename):
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    from textwrap import wrap

    # Read the data from the disk
    data = row.data.get('results-asr.shg.json')
    gap = row.get('gap_dir')
    atoms = row.toatoms()
    pbc = atoms.pbc.tolist()
    nd = np.sum(pbc)

    # Remove the files if it is already exist
    for fname in filename:
        if (Path(fname).is_file()):
            os.remove(fname)

    # Plot the data and add the axis labels
    sym_chi = data['symm']
    if len(sym_chi) == 1:
        raise CentroSymmetric
    chi = data['chi']

    if not chi:
        return
    w_l = data['freqs']
    fileind = 0
    axes = []

    for pol in sorted(chi.keys()):
        # Make the axis and add y=0 axis
        shg = chi[pol]
        ax = plt.figure().add_subplot(111)
        ax.axhline(y=0, color='k')

        # Add the bandgap
        bg = gap
        if bg is not None:
            ax.axvline(x=bg, color='k', ls='--')
            ax.axvline(x=bg / 2, color='k', ls='--')
            maxw = min(np.ceil(2.0 * bg), 7)
        else:
            maxw = 7

        # Plot the data
        amp_l = shg
        amp_l = amp_l[w_l < maxw]
        ax.plot(w_l[w_l < maxw], np.real(amp_l), '-', c='C0', label='Re')
        ax.plot(w_l[w_l < maxw], np.imag(amp_l), '-', c='C1', label='Im')
        ax.plot(w_l[w_l < maxw], np.abs(amp_l), '-', c='C2', label='Abs')

        # Set the axis limit
        ax.set_xlim(0, maxw)
        relation = sym_chi.get(pol)
        if not (relation is None):
            figtitle = '$' + '$\n$'.join(wrap(relation, 40)) + '$'
            ax.set_title(figtitle)
        ax.set_xlabel(r'Pump photon energy $\hbar\omega$ [eV]')
        polstr = f'{pol}'
        if nd == 2:
            ax.set_ylabel(r'$\chi^{(2)}_{' + polstr + r'}$ [nm$^2$/V]')
        else:
            ax.set_ylabel(r'$\chi^{(2)}_{' + polstr + r'}$ [nm/V]')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))

        # Add the legend
        ax.legend(loc='best')

        # Remove the extra space and save the figure
        plt.tight_layout()
        plt.savefig(filename[fileind])
        fileind += 1
        axes.append(ax)
        plt.close()

    # Now make the polarization resolved plot
    psi = np.linspace(0, 2 * np.pi, 201)
    selw = 0
    wind = np.argmin(np.abs(w_l - selw))
    if (Path('shgpol.npy').is_file()):
        os.remove('shgpol.npy')
    chipol = calc_polarized_shg(
        sym_chi, chi,
        wind=[wind], theta=0, phi=0,
        pte=np.sin(psi), ptm=np.cos(psi), outname=None, outbasis='pol')
    ax = plt.subplot(111, projection='polar')
    ax.plot(psi, np.abs(chipol[0]), 'C0', lw=1.0)
    ax.plot(psi, np.abs(chipol[1]), 'C1', lw=1.0)
    # Set the y limits
    ax.grid(True)
    rmax = np.amax(np.abs(chipol))
    if np.abs(rmax) < 1e-6:
        rmax = 1e-4
        ax.plot(0, 0, 'o', color='b', markersize=5)
    ax.set_rlim(0, 1.2 * rmax)
    ax.set_rgrids([rmax], fmt=r'%4.2g')
    labs = [r'  $\theta=0$', '45', '90', '135', '180', '225', '270', '315']
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315], labels=labs)

    # Put a legend below current axis
    ax.legend([r'Parallel: |$\chi^{(2)}_{\theta \theta \theta}$|',
               r'Perpendicular: |$\chi^{(2)}_{(\theta+90)\theta \theta}$|'],
              loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, ncol=2)

    # Remove the extra space and save the figure
    plt.tight_layout()
    plt.savefig(filename[fileind])
    axes.append(ax)

    return tuple(axes)


def calc_polarized_shg(sym_chi, chi_dict, wind=[1], theta=0.0, phi=0.0,
                       pte=[1.0], ptm=[0.0], E0=[1.0], outname=None,
                       outbasis='pol'):
    # Check the input arguments
    pte = np.array(pte)
    ptm = np.array(ptm)
    E0 = np.array(E0)
    assert np.all(
        np.abs(pte) ** 2 + np.abs(ptm) ** 2) == 1, \
        '|pte|**2+|ptm|**2 should be one.'
    assert len(pte) == len(ptm), 'Size of pte and ptm should be the same.'

    # Useful variables
    costh = np.cos(theta)
    sinth = np.sin(theta)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    nw = len(wind)
    npsi = len(pte)

    # Transfer matrix between (x y z)/(atm ate k) unit vectors basis
    if theta == 0:
        transmat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        transmat = [[cosphi * costh, sinphi * costh, -sinth],
                    [-sinphi, cosphi, 0],
                    [sinth * cosphi, sinth * sinphi, costh]]
    transmat = np.array(transmat)

    # Get the full chi tensor
    chi_vvvl = make_full_chi(sym_chi, chi_dict)

    # Check the E0
    if len(E0) == 1:
        E0 = E0 * np.ones((nw))

    # in xyz coordinate
    Einc = np.zeros((3, npsi), dtype=complex)
    for v1 in range(3):
        Einc[v1] = (pte * transmat[0][v1] + ptm * transmat[1][v1])

    # Loop over components
    chipol = np.zeros((3, npsi, nw), dtype=complex)
    for ii, wi in enumerate(wind):
        for ind in range(27):
            v1, v2, v3 = int(ind / 9), int((ind % 9) / 3), (ind % 9) % 3
            if chi_vvvl[v1, v2, v3, wi] != 0.0:
                chipol[v1, :, ii] += chi_vvvl[v1, v2, v3, wi] * \
                    Einc[v2, :] * Einc[v3, :] * E0[ii]**2

    # Change the output basis if needed, and return
    if outbasis == 'xyz':
        chipol_new = chipol
    elif outbasis == 'pol':
        chipol_new = np.zeros((3, npsi, nw), dtype=complex)
        for ind, wi in enumerate(wind):
            chipol[:, :, ind] = np.dot(transmat.T, chipol[:, :, ind])
            chipol_new[0, :, ind] = chipol[0, :, ind] * \
                pte + chipol[1, :, ind] * ptm
            chipol_new[1, :, ind] = -chipol[0, :, ind] * \
                ptm + chipol[1, :, ind] * pte

    else:
        raise NotImplementedError

    return chipol_new

def ShgWebpanel(result, row, key_descriptions):
    from asr.database.browser import (fig)
    from textwrap import wrap

    # Get the data
    data = row.data.get('results-asr.shg.json')
    if data is None:
        return

    # Make the table
    sym_chi = data.get('symm')
    table = []
    for pol in sorted(sym_chi.keys()):
        relation = sym_chi[pol]
        if pol == 'zero':
            if relation != '':
                pol = 'Others'
                relation = '0=' + relation
            else:
                continue

        if (len(relation) == 3):
            relation_new = ''
        else:
            # relation_new = '$'+'$\n$'.join(wrap(relation, 40))+'$'
            relation_new = '\n'.join(wrap(relation, 50))
        table.append((pol, relation_new))
    opt = {'type': 'table',
           'header': ['Element', 'Relations'],
           'rows': table}

    # Make the figure list
    npan = len(sym_chi)
    files = ['shg{}.png'.format(ii + 1) for ii in range(npan)]
    cols = [[fig(f'shg{2 * ii + 1}.png'),
             fig(f'shg{2 * ii + 2}.png')] for ii in range(int(npan / 2))]
    if npan % 2 == 0:
        cols.append([opt, None])
    else:
        cols.append([fig(f'shg{npan}.png'), opt])
    # Transpose the list
    cols = np.array(cols).T.tolist()

    panel = {'title': 'SHG spectrum (RPA)',
             'columns': cols,
             'plot_descriptions':
                 [{'function': plot_shg,
                   'filenames': files}],
             'sort': 20}

    return [panel]


@prepare_result
class ShgResult(ASRResult):

    freqs: typing.List[float]
    chi: typing.Dict[str, typing.Any]
    symm: typing.Dict[str, str]

    key_descriptions = {
        "freqs": "Pump photon energy [eV]",
        "chi": "Non-zero SHG tensor elements in SI units",
        "symm": "Symmetry relation of SHG tensor",
    }
    formats = {"ase_webpanel": ShgWebpanel}


######### shift #########
def ShiftWebpanel(result, row, key_descriptions):
    from asr.database.browser import (fig)
    from textwrap import wrap

    # Get the data
    data = row.data.get('results-asr.shift.json')

    # Make the table
    sym_chi = data.get('symm')
    table = []
    for pol in sorted(sym_chi.keys()):
        relation = sym_chi[pol]
        if pol == 'zero':
            if relation != '':
                pol = 'Others'
                relation = '0=' + relation
            else:
                continue

        if (len(relation) == 3):
            relation_new = ''
        else:
            # relation_new = '$'+'$\n$'.join(wrap(relation, 40))+'$'
            relation_new = '\n'.join(wrap(relation, 50))
        table.append((pol, relation_new))
    opt = {'type': 'table',
           'header': ['Element', 'Relations'],
           'rows': table}

    # Make the figure list
    npan = len(sym_chi) - 1
    files = ['shift{}.png'.format(ii + 1) for ii in range(npan)]
    cols = [[fig(f'shift{2 * ii + 1}.png'),
             fig(f'shift{2 * ii + 2}.png')] for ii in range(int(npan / 2))]
    if npan % 2 == 0:
        cols.append([opt, None])
    else:
        cols.append([fig(f'shift{npan}.png'), opt])
    # Transpose the list
    cols = np.array(cols).T.tolist()

    panel = {'title': 'Shift current spectrum (RPA)',
             'columns': cols,
             'plot_descriptions':
                 [{'function': plot_shift,
                   'filenames': files}],
             'sort': 20}

    return [panel]


@prepare_result
class ShiftResult(ASRResult):

    freqs: typing.List[float]
    sigma: typing.Dict[str, typing.Any]
    symm: typing.Dict[str, str]

    key_descriptions = {
        "freqs": "Photon energy [eV]",
        "sigma": "Non-zero shift conductivity tensor elements in SI units",
        "symm": "Symmetry relation of shift conductivity tensor",
    }
    formats = {"ase_webpanel": ShiftWebpanel}


def plot_shift(row, *filename):
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    from textwrap import wrap

    # Read the data from the disk
    data = row.data.get('results-asr.shift.json')
    gap = row.get('gap_dir_nosoc')
    atoms = row.toatoms()
    pbc = atoms.pbc.tolist()
    nd = np.sum(pbc)
    if data is None:
        return

    # Remove the files if it is already exist
    for fname in filename:
        if (Path(fname).is_file()):
            os.remove(fname)

    # Plot the data and add the axis labels
    sym_chi = data['symm']
    if len(sym_chi) == 1:
        raise CentroSymmetric
    sigma = data['sigma']

    if not sigma:
        return
    w_l = data['freqs']
    fileind = 0
    axes = []

    for pol in sorted(sigma.keys()):
        # Make the axis and add y=0 axis
        shift_l = sigma[pol]
        ax = plt.figure().add_subplot(111)
        ax.axhline(y=0, color='k')

        # Add the bandgap
        if gap is not None:
            ax.axvline(x=gap, color='k', ls='--')

        # Plot the data
        ax.plot(w_l, np.real(shift_l), '-', c='C0',)

        # Set the axis limit
        ax.set_xlim(0, np.max(w_l))
        relation = sym_chi.get(pol)
        if not (relation is None):
            figtitle = '$' + '$\n$'.join(wrap(relation, 40)) + '$'
            ax.set_title(figtitle)
        ax.set_xlabel(r'Energy [eV]')
        polstr = f'{pol}'
        if nd == 2:
            ax.set_ylabel(r'$\sigma^{(2)}_{' + polstr + r'}$ [nm$\mu$A/V$^2$]')
        else:
            ax.set_ylabel(r'$\sigma^{(2)}_{' + polstr + r'} [$\mu$A/V$^2$]')
        ax.ticklabel_format(axis='both', style='plain', scilimits=(-2, 2))

        # Remove the extra space and save the figure
        plt.tight_layout()
        plt.savefig(filename[fileind])
        fileind += 1
        axes.append(ax)
        plt.close()

    return tuple(axes)


######### sj_analyze #########
panel_description = make_panel_description(
    """
Analysis of the thermodynamic stability of the defect using Slater-Janak
 transition state theory.
""",
    articles=[
        href("""M. Pandey et al. Defect-tolerant monolayer transition metal
dichalcogenides, Nano Letters, 16 (4) 2234 (2016)""",
             'https://doi.org/10.1021/acs.nanolett.5b04513'),
    ],
)

def SJAnalyzeWebpanel(result, row, key_descriptions):

    explained_keys = []
    for key in ['eform']:
        if key in result.key_descriptions:
            key_description = result.key_descriptions[key]
            explanation = key_description
            explained_key = describe_entry(key, description=explanation)
        else:
            explained_key = key
        explained_keys.append(explained_key)

    defname = row.defect_name
    defstr = f"{defname.split('_')[0]}<sub>{defname.split('_')[1]}</sub>"
    formation_table_sum = get_summary_table(result)
    formation_table = get_formation_table(result, defstr)
    # defectinfo = row.data.get('asr.defectinfo.json')
    transition_table = get_transition_table(result, defstr)

    panel = WebPanel(
        describe_entry('Formation energies and charge transition levels (Slater-Janak)',
                       panel_description),
        columns=[[describe_entry(fig('sj_transitions.png'),
                                 'Slater-Janak calculated charge transition levels.'),
                  transition_table],
                 [describe_entry(fig('formation.png'),
                                 'Formation energy diagram.'),
                  formation_table]],
        plot_descriptions=[{'function': plot_charge_transitions,
                            'filenames': ['sj_transitions.png']},
                           {'function': plot_formation_energies,
                            'filenames': ['formation.png']}],
        sort=29)

    summary = {'title': 'Summary',
               'columns': [[formation_table_sum],
                           []],
               'sort': 0}

    return [panel, summary]


@prepare_result
class PristineResults(ASRResult):
    """Container for pristine band gap results."""

    vbm: float
    cbm: float
    evac: float

    key_descriptions = dict(
        vbm='Pristine valence band maximum [eV].',
        cbm='Pristine conduction band minimum [eV]',
        evac='Pristine vacuum level [eV]')


@prepare_result
class TransitionValues(ASRResult):
    """Container for values of a specific charge transition level."""

    transition: float
    erelax: float
    evac: float

    key_descriptions = dict(
        transition='Charge transition level [eV]',
        erelax='Reorganization contribution  to the transition level [eV]',
        evac='Vacuum level for halfinteger calculation [eV]')


@prepare_result
class TransitionResults(ASRResult):
    """Container for charge transition level results."""

    transition_name: str
    transition_values: TransitionValues

    key_descriptions = dict(
        transition_name='Name of the charge transition (Initial State/Final State)',
        transition_values='Container for values of a specific charge transition level.')


@prepare_result
class TransitionListResults(ASRResult):
    """Container for all charge transition level results."""

    transition_list: typing.List[TransitionResults]

    key_descriptions = dict(
        transition_list='List of TransitionResults objects.')


@prepare_result
class StandardStateResult(ASRResult):
    """Container for results related to the standard state of the present defect."""

    element: str
    eref: float

    key_descriptions = dict(
        element='Atomic species.',
        eref='Reference energy extracted from OQMD [eV].')


@prepare_result
class SJAnalyzeResult(ASRResult):
    """Container for Slater Janak results."""

    transitions: typing.List[TransitionResults]
    pristine: PristineResults
    eform: typing.List[typing.Tuple[float, int]]
    standard_states: typing.List[StandardStateResult]
    hof: float

    key_descriptions = dict(
        transitions='Charge transition levels with [transition energy, '
                    'relax correction, reference energy] eV',
        pristine='Container for pristine band gap results.',
        eform='List of formation energy tuples (eform wrt. standard states [eV], '
              'charge state)',
        standard_states='List of StandardStateResult objects for each species.',
        hof='Heat of formation for the pristine monolayer [eV]')

    formats = {"ase_webpanel": SJAnalyzeWebpanel}


def f(x, a, b):
    return a * x + b


def get_formation_table(result, defstr):
    from asr.database.browser import table, describe_entry

    formation_table = table(result, 'Defect formation energy', [])
    for element in result.eform:
        formation_table['rows'].extend(
            [[describe_entry(f'{defstr} (q = {element[1]:1d} @ VBM)',
                             description='Formation energy for charge state q '
                                         'at the valence band maximum [eV].'),
              f'{element[0]:.2f} eV']])

    return formation_table


def plot_formation_energies(row, fname):
    """Plot formation energies and transition levels within the gap."""
    import matplotlib.pyplot as plt

    colors = {'0': 'C0',
              '1': 'C1',
              '2': 'C2',
              '3': 'C3',
              '-1': 'C4',
              '-2': 'C5',
              '-3': 'C6',
              '-4': 'C7',
              '4': 'C8'}

    data = row.data.get('results-asr.sj_analyze.json')

    vbm = data['pristine']['vbm']
    cbm = data['pristine']['cbm']
    gap = abs(cbm - vbm)
    eform = data['eform']
    transitions = data['transitions']

    fig, ax1 = plt.subplots()

    ax1.axvspan(-20, 0, color='grey', alpha=0.5)
    ax1.axvspan(gap, 20, color='grey', alpha=0.5)
    ax1.axhline(0, color='black', linestyle='dotted')
    ax1.axvline(gap, color='black', linestyle='solid')
    ax1.axvline(0, color='black', linestyle='solid')
    for element in eform:
        ax1.plot([0, gap], [f(0, element[1], element[0]),
                            f(gap, element[1], element[0])],
                 color=colors[str(element[1])],
                 label=element[1])

    ax1.set_xlim(-0.2 * gap, gap + 0.2 * gap)
    yrange = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    ax1.text(-0.1 * gap, 0.5 * yrange, 'VBM', ha='center',
             va='center', rotation=90, weight='bold', color='white',
             fontsize=12)
    ax1.text(gap + 0.1 * gap, 0.5 * yrange, 'CBM', ha='center',
             va='center', rotation=90, weight='bold', color='white',
             fontsize=12)

    tickslist = []
    labellist = []
    energies = []
    for i, element in enumerate(transitions):
        name = element['transition_name']
        q = int(name.split('/')[-1])
        if q < 0:
            energy = (element['transition_values']['transition']
                      - element['transition_values']['erelax']
                      - element['transition_values']['evac'] - vbm)
        elif q > 0:
            energy = (element['transition_values']['transition']
                      + element['transition_values']['erelax']
                      - element['transition_values']['evac'] - vbm)
        energies.append(energy)
        if energy > 0 and energy < (gap):
            tickslist.append(energy)
            labellist.append(name)
            ax1.axvline(energy, color='grey', linestyle='dotted')
    energies.append(100)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([])
    ax1.set_xlabel(r'$E_\mathrm{F} - E_\mathrm{VBM}}$ [eV]')
    ax1.set_ylabel(r'$E^f$ (wrt. standard states) [eV]')
    ax1.legend()

    plt.tight_layout()

    plt.savefig(fname)
    plt.close()


def plot_charge_transitions(row, fname):
    """Plot calculated CTL along with the pristine bandgap."""
    import matplotlib.pyplot as plt

    colors = {'0': 'C0',
              '1': 'C1',
              '2': 'C2',
              '3': 'C3',
              '-1': 'C4',
              '-2': 'C5',
              '-3': 'C6',
              '-4': 'C7',
              '4': 'C8'}

    data = row.data.get('results-asr.sj_analyze.json')

    vbm = data['pristine']['vbm']
    cbm = data['pristine']['cbm']

    gap = abs(cbm - vbm)

    transitions = data['transitions']

    plt.xlim(-1, 1)
    plt.ylim(-0.2 * gap, gap + 0.2 * gap)
    plt.xticks([], [])

    plt.axhspan(-5, 0, color='grey', alpha=0.5)
    plt.axhspan(gap, gap + 5, color='grey', alpha=0.5)
    plt.axhline(0, color='black', linestyle='solid')
    plt.axhline(gap, color='black', linestyle='solid')
    plt.text(0, -0.1 * gap, 'VBM', color='white',
             ha='center', va='center', weight='bold',
             fontsize=12)
    plt.text(0, gap + 0.1 * gap, 'CBM', color='white',
             ha='center', va='center', weight='bold',
             fontsize=12)

    i = 1
    for trans in transitions:
        name = trans['transition_name']
        q = int(name.split('/')[-1])
        q_new = int(name.split('/')[0])
        if q > 0:
            y = (trans['transition_values']['transition']
                 + trans['transition_values']['erelax']
                 - trans['transition_values']['evac'])
            color1 = colors[str(q)]
            color2 = colors[str(q_new)]
        elif q < 0:
            y = (trans['transition_values']['transition']
                 - trans['transition_values']['erelax']
                 - trans['transition_values']['evac'])
            color1 = colors[str(q)]
            color2 = colors[str(q_new)]
        if y <= (cbm + 0.2 * gap) and y >= (vbm - 0.2 * gap):
            plt.plot(np.linspace(-0.9, 0.5, 20), 20 * [y - vbm],
                     label=trans['transition_name'],
                     color=color1, mec=color2, mfc=color2, marker='s', markersize=3)
            i += 1

    plt.legend(loc='center right')
    plt.ylabel(r'$E - E_{\mathrm{VBM}}$ [eV]')
    plt.yticks()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


######### stiffness #########
panel_description = make_panel_description(
    """
The stiffness tensor (C) is a rank-4 tensor that relates the stress of a
material to the applied strain. In Voigt notation, C is expressed as a NxN
matrix relating the N independent components of the stress and strain
tensors. C is calculated as a finite difference of the stress under an applied
strain with full relaxation of atomic coordinates. A negative eigenvalue of C
indicates a dynamical instability.
""",
    articles=['C2DB'],
)


def StiffnessWebpanel(result, row, key_descriptions):
    import numpy as np

    stiffnessdata = row.data['results-asr.stiffness.json']
    c_ij = stiffnessdata['stiffness_tensor'].copy()
    eigs = stiffnessdata['eigenvalues'].copy()
    nd = np.sum(row.pbc)

    if nd == 2:
        c_ij = np.zeros((4, 4))
        c_ij[1:, 1:] = stiffnessdata['stiffness_tensor']
        ctable = matrixtable(
            stiffnessdata['stiffness_tensor'],
            title='C<sub>ij</sub> (N/m)',
            columnlabels=['xx', 'yy', 'xy'],
            rowlabels=['xx', 'yy', 'xy'])

        eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']]
                   + [[f'Eigenvalue {ie}', f'{eig.real:.2f} N/m']
                      for ie, eig in enumerate(sorted(eigs,
                                                      key=lambda x: x.real))])
    elif nd == 3:
        eigs *= 1e-9
        c_ij *= 1e-9
        ctable = matrixtable(
            c_ij,
            title='C<sub>ij</sub> (10⁹ N/m²)',
            columnlabels=['xx', 'yy', 'zz', 'yz', 'xz', 'xy'],
            rowlabels=['xx', 'yy', 'zz', 'yz', 'xz', 'xy'])

        eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']]
                   + [[f'Eigenvalue {ie}', f'{eig.real:.2f} · 10⁹ N/m²']
                      for ie, eig
                      in enumerate(sorted(eigs, key=lambda x: x.real))])
    else:
        ctable = dict(
            type='table',
            rows=[])
        eig = complex(eigs[0])
        eigrows = ([['<b>Stiffness tensor eigenvalues<b>', '']]
                   + [[f'Eigenvalue', f'{eig.real:.2f} * 10⁻¹⁰ N']])

    eigtable = dict(
        type='table',
        rows=eigrows)

    panel = {'title': describe_entry('Stiffness tensor',
                                     description=panel_description),
             'columns': [[ctable], [eigtable]],
             'sort': 2}

    return [panel]


@prepare_result
class StiffnessResult(ASRResult):

    c_11: float
    c_12: float
    c_13: float
    c_14: float
    c_15: float
    c_16: float
    c_21: float
    c_22: float
    c_23: float
    c_24: float
    c_25: float
    c_26: float
    c_31: float
    c_32: float
    c_33: float
    c_34: float
    c_35: float
    c_36: float
    c_41: float
    c_42: float
    c_43: float
    c_44: float
    c_45: float
    c_46: float
    c_51: float
    c_52: float
    c_53: float
    c_54: float
    c_55: float
    c_56: float
    c_61: float
    c_62: float
    c_63: float
    c_64: float
    c_65: float
    c_66: float

    __links__: typing.List[str]

    stiffness_tensor: typing.List[typing.List[float]]
    eigenvalues: typing.List[complex]
    dynamic_stability_stiffness: str
    speed_of_sound_x: float
    speed_of_sound_y: float

    key_descriptions = {
        "c_11": "Stiffness tensor 11-component.",
        "c_12": "Stiffness tensor 12-component.",
        "c_13": "Stiffness tensor 13-component.",
        "c_14": "Stiffness tensor 14-component.",
        "c_15": "Stiffness tensor 15-component.",
        "c_16": "Stiffness tensor 16-component.",
        "c_21": "Stiffness tensor 21-component.",
        "c_22": "Stiffness tensor 22-component.",
        "c_23": "Stiffness tensor 23-component.",
        "c_24": "Stiffness tensor 24-component.",
        "c_25": "Stiffness tensor 25-component.",
        "c_26": "Stiffness tensor 26-component.",
        "c_31": "Stiffness tensor 31-component.",
        "c_32": "Stiffness tensor 32-component.",
        "c_33": "Stiffness tensor 33-component.",
        "c_34": "Stiffness tensor 34-component.",
        "c_35": "Stiffness tensor 35-component.",
        "c_36": "Stiffness tensor 36-component.",
        "c_41": "Stiffness tensor 41-component.",
        "c_42": "Stiffness tensor 42-component.",
        "c_43": "Stiffness tensor 43-component.",
        "c_44": "Stiffness tensor 44-component.",
        "c_45": "Stiffness tensor 45-component.",
        "c_46": "Stiffness tensor 46-component.",
        "c_51": "Stiffness tensor 51-component.",
        "c_52": "Stiffness tensor 52-component.",
        "c_53": "Stiffness tensor 53-component.",
        "c_54": "Stiffness tensor 54-component.",
        "c_55": "Stiffness tensor 55-component.",
        "c_56": "Stiffness tensor 56-component.",
        "c_61": "Stiffness tensor 61-component.",
        "c_62": "Stiffness tensor 62-component.",
        "c_63": "Stiffness tensor 63-component.",
        "c_64": "Stiffness tensor 64-component.",
        "c_65": "Stiffness tensor 65-component.",
        "c_66": "Stiffness tensor 66-component.",
        "eigenvalues": "Stiffness tensor eigenvalues.",
        "speed_of_sound_x": "Speed of sound (x) [m/s]",
        "speed_of_sound_y": "Speed of sound (y) [m/s]",
        "stiffness_tensor": "Stiffness tensor [`N/m^{dim-1}`]",
        "dynamic_stability_stiffness": "Stiffness dynamic stability (low/high)",
        "__links__": "UIDs to strained folders."
    }

    formats = {"ase_webpanel": StiffnessWebpanel}


######### structureinfo #########
dynstab_description = """\
Dynamically stable materials are stable against small perturbations of
their structure (atom positions and unit cell shape). The structure
thus represents a local minimum on the {PES}.

DS materials are characterised by having only real, non-negative {phonon}
frequencies and positive definite {stiffnesstensor}.
""".format(
    PES=href('potential energy surface',
             'https://en.wikipedia.org/wiki/Potential_energy_surface'),
    phonon=href('phonon', 'https://en.wikipedia.org/wiki/Phonon'),
    stiffnesstensor=href(
        'stiffness tensor',
        'https://en.wikiversity.org/wiki/Elasticity/Constitutive_relations'))
# XXX This string is used in CMR also in search.html also.
# It should probably be imported from here instead.


def StructureInfoWebpanel(result, row, key_descriptions):
    from asr.database.browser import describe_entry, href, table

    spglib = get_spg_href('https://spglib.github.io/spglib/')
    crystal_type = describe_crystaltype_entry(spglib)

    spg_list_link = href(
        'Space group', 'https://en.wikipedia.org/wiki/List_of_space_groups'
    )

    layergroup_link = href(
        'Layer group', 'https://en.wikipedia.org/wiki/Layer_group')

    spacegroup = describe_entry(
        'spacegroup',
        f"{spg_list_link} determined with {spglib}."
        f"The {spg_list_link} determined with {spglib} by stacking the "
        f"monolayer in A-A configuration."
    )

    spgnum = describe_entry(
        'spgnum',
        f"{spg_list_link} number determined with {spglib}."
        f"{spg_list_link} number determined with {spglib} by stacking the "
        f"monolayer in A-A configuration."
    )

    layergroup = describe_entry(
        'layergroup',
        f'{layergroup_link} determined with {spglib}')
    lgnum = describe_entry(
        'lgnum',
        f'{layergroup_link} number determined with {spglib}')

    pointgroup = describe_pointgroup_entry(spglib)

    icsd_link = href('Inorganic Crystal Structure Database (ICSD)',
                     'https://icsd.products.fiz-karlsruhe.de/')

    icsd_id = describe_entry(
        'icsd_id',
        f"ID of a closely related material in the {icsd_link}."
    )

    cod_link = href(
        'Crystallography Open Database (COD)',
        'http://crystallography.net/cod/browse.html'
    )

    cod_id = describe_entry(
        'cod_id',
        f"ID of a closely related material in the {cod_link}."
    )

    # Here we are hacking the "label" out of a row without knowing
    # whether there is a label, or that the "label" recipe exists.

    tablerows = [
        crystal_type, layergroup, lgnum, spacegroup, spgnum, pointgroup,
        icsd_id, cod_id]

    # The table() function is EXTREMELY illogical.
    # I can't get it to work when appending another row
    # to the tablerows list.  Therefore we append rows afterwards.  WTF.
    basictable = table(row, 'Structure info', tablerows, key_descriptions, 2)
    rows = basictable['rows']

    labelresult = row.data.get('results-asr.c2db.labels.json')
    if labelresult is not None:
        tablerow = labelresult.as_formatted_tablerow()
        rows.append(tablerow)

    codid = row.get('cod_id')
    if codid:
        # Monkey patch to make a link
        for tmprow in rows:
            href = ('<a href="http://www.crystallography.net/cod/'
                    + '{id}.html">{id}</a>'.format(id=codid))
            if 'cod_id' in tmprow[0]:
                tmprow[1] = href

    doi = row.get('doi')
    doistring = describe_entry(
        'Reported DOI',
        'DOI of article reporting the synthesis of the material.'
    )
    if doi:
        rows.append([
            doistring,
            '<a href="https://doi.org/{doi}" target="_blank">{doi}'
            '</a>'.format(doi=doi)
        ])

    # There should be a central place defining "summary" panel to take
    # care of stuff that comes not from an individual "recipe" but
    # from multiple ones.  Such as stability, or listing multiple band gaps
    # next to each other, etc.
    #
    # For now we stick this in structureinfo but that is god-awful.
    phonon_stability = row.get('dynamic_stability_phonons')
    stiffness_stability = row.get('dynamic_stability_stiffness')

    from asr.paneldata import ehull_table_rows
    ehull_table_rows = ehull_table_rows(row, key_descriptions)['rows']

    if phonon_stability is not None and stiffness_stability is not None:
        # XXX This will easily go wrong if 'high'/'low' strings are changed.
        dynamically_stable = (
            phonon_stability == 'high' and stiffness_stability == 'high')

        yesno = ['No', 'Yes'][dynamically_stable]

        dynstab_row = [describe_entry('Dynamically stable', dynstab_description), yesno]
        dynstab_rows = [dynstab_row]
    else:
        dynstab_rows = []

    panel = {'title': 'Summary',
             'columns': [[basictable,
                          {'type': 'table', 'header': ['Stability', ''],
                           'rows': [*ehull_table_rows, *dynstab_rows]}],
                         [{'type': 'atoms'}, {'type': 'cell'}]],
             'sort': -1}

    return [panel]


@prepare_result
class StructureInfoResult(ASRResult):

    cell_area: float
    has_inversion_symmetry: bool
    stoichiometry: str
    spacegroup: str
    spgnum: int
    layergroup: str
    lgnum: int
    pointgroup: str
    crystal_type: str
    spglib_dataset: dict
    formula: str

    key_descriptions = {
        "cell_area": "Area of unit-cell [`Å²`]",
        "has_inversion_symmetry": "Material has inversion symmetry",
        "stoichiometry": "Stoichiometry",
        "spacegroup": "Space group (AA stacking)",
        "spgnum": "Space group number (AA stacking)",
        "layergroup": "Layer group",
        "lgnum": "Layer group number",
        "pointgroup": "Point group",
        "crystal_type": "Crystal type",
        "spglib_dataset": "SPGLib symmetry dataset.",
        "formula": "Chemical formula."
    }

    formats = {"ase_webpanel": StructureInfoWebpanel}


def get_spg_href(url):
    return href('SpgLib', url)


def describe_crystaltype_entry(spglib):
    crystal_type = describe_entry(
        'Crystal type',
        "The crystal type is defined as "
        + br
        + div(bold('-'.join([code('stoi'),
                             code('spg no.'),
                             code('occ. wyck. pos.')])), 'well well-sm text-center')
        + 'where'
        + dl(
            [
                [code('stoi'), 'Stoichiometry.'],
                [code('spg no.'), f'The space group calculated with {spglib}.'],
                [code('occ. wyck. pos.'),
                 'Alphabetically sorted list of occupied '
                 f'wyckoff positions determined with {spglib}.'],
            ]
        )
    )

    return crystal_type


def describe_pointgroup_entry(spglib):
    pointgroup = describe_entry(
        'Point group',
        f"Point group determined with {spglib}."
    )

    return pointgroup


######### tdm #########
@prepare_result
class TdmResult(ASRResult):
    """Container for transition dipole moment results."""

    d_snnv: typing.List[np.ndarray]
    n1: int
    n2: int

    key_descriptions = dict(
        d_snnv='transition dipole matrix elements.',
        n1='staterange minimum.',
        n2='staterange maximum.')


######### zfs #########
def get_zfs_table(result):
    zfs_array = np.zeros((2, 3))
    rowlabels = ['Spin 0', 'Spin 1']
    for i, element in enumerate(zfs_array):
        for j in range(3):
            zfs_array[i, j] = result['D_vv'][i][j]

    rows = []
    for i in range(len(zfs_array)):
        rows.append((rowlabels[i],
                     f'{zfs_array[i][0]:.2f} MHz',
                     f'{zfs_array[i][1]:.2f} MHz',
                     f'{zfs_array[i][2]:.2f} MHz'))

    zfs_table = {'type': 'table',
                 'header': ['Spin channel',
                            'D<sub>xx</sub>',
                            'D<sub>yy</sub>',
                            'D<sub>zz</sub>']}
    zfs_table['rows'] = rows

    return zfs_table


def ZfsWebpanel(result, row, key_description):
    zfs_table = get_zfs_table(result)
    zfs = WebPanel('Zero field splitting (ZFS)',
                   columns=[[], [zfs_table]],
                   sort=41)

    return [zfs]


@prepare_result
class ZfsResult(ASRResult):
    """Container for zero-field-splitting results."""

    D_vv: np.ndarray

    key_descriptions = dict(
        D_vv='Zero-field-splitting components for each spin channel '
             'and each direction (x, y, z) [MHz].')

    formats = {'ase_webpanel': ZfsWebpanel}
