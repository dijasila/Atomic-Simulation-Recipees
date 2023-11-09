import itertools
import typing
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from ase import Atoms
from ase.formula import Formula
from ase.db.row import AtomsRow
from ase.dft.kpoints import BandPath
from ase.dft.kpoints import labels_from_kpts

from asr.core import ASRResult, prepare_result
from asr.utils.hacks import gs_xcname_from_row
from asr.panels.createwebpanel import (
    GsWebpanel,
    BaderWebpanel, BornChargesWebpanel, ChargeNeutralityWebpanel,
    BandstructureWebpanel, ProjBSWebpanel, DOSWebpanel, EmassesWebpanel,
    FermiWebpanel, PdosWebpanel,
    BerryWebpanel, ExchangeWebpanel, HFWebpanel, MagAniWebpanel,
    MagStateWebpanel, SJAnalyzeWebpanel, ZfsWebpanel,
    CHCWebpanel, ConvexHullWebpanel,
    DefectSymmetryWebpanel, DefectInfoWebpanel, DefectLinksWebpanel,
    DefPotsWebpanel, StiffnessWebpanel, StructureInfoWebpanel,
    PhononWebpanel, PhonopyWebpanel,
    PiezoEleTenWebpanel,
)
from asr.extra_result_plotting import (
    add_fermi, plot_with_colors, legend_on_top,
)


@dataclass
class BaseResult:

    @classmethod
    def from_row(self, row: AtomsRow, rowkey: str) -> 'BaseResult':
        ...

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_json(cls, string: str) -> object:
        """
        Recreate the object from a string specifying the path to a file or a
        string representation of a json file. E.g., json_str=json.dumps(obj).
        """
        try:
            dct = json.loads(string)
        except:
            with open(string, "r") as file:
                dct = json.load(file)
        return cls(**dct)

    def to_json(self, filename: [str, bool] = False):
        json_str = json.dumps(self.__dict__)

        if filename:
            with open(filename, "w") as file:
                file.write(json_str)
        else:
            return json_str

    def to_pandas(self, keys: list = []):
        """
        A result must know what data to put into a table.
        USE PANDAS!!!!
        """
        keys = self.__dict__ if len(keys) == 0 else keys

        filtered_data = {key: self[key] for key in keys}

        df = pd.DataFrame.from_dict(filtered_data, orient='index')

        return df

    def __getitem__(self, item):
        return self.__dict__[item]

    def get_plot(self):
        """
        A result should know how to plot itself!
        """
        pass


class CompositeResult:
    pass


## Charges
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
        from asr.some_row_manipulation_garbage import get_overview_tables
        return get_overview_tables(scresult, result, unitstring)

    @staticmethod
    def get_conc_table(result, element, unitstring):
        from asr.some_row_manipulation_garbage import get_conc_table
        return get_conc_table(result, element, unitstring)

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


## Electronic Structure
# Bandstructure
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
            line=dict(color='rgb(0, 0, 0)', width=2, dash='dash'),
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
        # XXX plot_with_colors never uses emin and emax. this input is
        # useless to pass.
        ax, cbar = plot_with_colors(
            bsp,
            ax=ax,
            energies=e_mk - ref_soc,
            colors=sz_mk,
            colorbar=colorbar,
            filename=filename,
            show=False,
            # emin=emin - ref_soc,
            # emax=emax - ref_soc,
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
# projected_bandstructure
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

    @staticmethod
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
        from ase.spectrum.band_structure import BandStructure, \
            BandStructurePlot
        from asr.extra_result_plotting import (
            get_yl_ordering, get_bs_sampling, get_pie_slice, get_pie_markers
        )

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
        plt.legend(legend_markers,
                   [yl.replace(',', ' (') + ')' for yl in yl_i],
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
# dos
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

    @staticmethod
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
Result = DOSResult  # backwards compatibility with old result files
# pdos
from asr.randomresults import PdosResult
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

    @staticmethod
    def plot_pdos_nosoc(*args, **kwargs):
        from asr.extra_result_plotting import plot_pdos
        return plot_pdos(*args, soc=False, **kwargs)
# Emasses
from asr.extra_result_plotting import plot_fit, plot_band, get_plot_data
from asr.some_row_manipulation_garbage import convert_key_to_tuple
MAXMASS = 10  # More that 90% of masses are less than this
class EmassesResult(ASRResult):
    pass
@prepare_result
class ValidateResult(ASRResult):

    formats = {"ase_webpanel": EmassesWebpanel}

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
# result processing should somehow live on the results to access data and
# process it.
def mareformat(mare):
    return str(round(mare, 3)) + " %"
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
def check_soc(spin_band_dict):
    for k in spin_band_dict.keys():
        if 'effmass' in k and 'nosoc' in k:
            return False

    return True
# fermisurface
@prepare_result
class FermiSurfaceResult(ASRResult):

    contours: list
    key_descriptions = {'contours': 'List of Fermi surface contours.'}

    formats = {"ase_webpanel": FermiWebpanel}
    @staticmethod
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
                direction = anis['spin_axis']
            else:
                direction = 'z'

            cbar = plt.colorbar()
            cbar.set_label(rf'$\langle S_{direction}\rangle/\hbar$', size=16)

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
# exchange
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
# hyperfine
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
    'V': (51, 11.21232),
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
    'I': (127, 8.56477221),
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
from asr.randomresults import HyperfineResult, GyromagneticResult
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

    @staticmethod
    def get_atoms_close_to_center(center, atoms):
        """
        Return ordered list of the atoms closest to the defect.

        Note, that this is the case only if a previous defect calculation is present.
        Return list of atoms closest to the origin otherwise.
        """
        from ase.geometry import get_distances
        _, distances = get_distances(center, atoms.positions, cell=atoms.cell,
                                     pbc=atoms.pbc)
        args = np.argsort(distances[0])

        return args, distances[0][args]

    @staticmethod
    def get_hf_table(hf_results, ordered_args):
        hf_array = np.zeros((10, 4))
        hf_atoms = []
        for i, arg in enumerate(ordered_args[:10]):
            hf_result = hf_results[arg]
            hf_atoms.append(
                hf_result['kind'] + ' (' + str(hf_result['index']) + ')')
            for j, value in enumerate(
                [hf_result['magmom'], *hf_result['eigenvalues']]):
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

    @staticmethod
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
# magnetic_anisotropy
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
# magstate
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
# orbmag
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
# sj_analyze
from asr.randomresults import (
    PristineResults, TransitionResults, StandardStateResult
)
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

    @staticmethod
    def get_formation_table(result, defstr):
        from asr.some_row_manipulation_garbage import get_formation_table
        return get_formation_table(result, defstr)

    @staticmethod
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

        from asr.extra_result_plotting import f
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
            if energy > 0 and energy < gap:
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

    @staticmethod
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
                         color=color1, mec=color2, mfc=color2, marker='s',
                         markersize=3)
                i += 1

        plt.legend(loc='center right')
        plt.ylabel(r'$E - E_{\mathrm{VBM}}$ [eV]')
        plt.yticks()
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
# tdm
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
# zfs
@prepare_result
class ZfsResult(ASRResult):
    """Container for zero-field-splitting results."""

    D_vv: np.ndarray

    key_descriptions = dict(
        D_vv='Zero-field-splitting components for each spin channel '
             'and each direction (x, y, z) [MHz].')

    formats = {'ase_webpanel': ZfsWebpanel}

    @staticmethod
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


## Convex Hull
from extra_result_plotting import (get_hull_energies, ObjectHandler)
# chc
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

    # XXX is this actually accessing self?? why is this called dct if it is
    # self
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

    # XXX is this actually accessing self?? why is this called dct if it is
    # self
    def from_dict(dct):
        mat_ref = Reference.from_dict(dct["mat_ref"])
        react_ref = Reference.from_dict(dct["react_ref"])
        ref = Reference.from_dict(dct["ref"])

        return LeanIntermediate(mat_ref, react_ref, ref)
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

    @staticmethod
    def chcut_plot(row, fname):
        import matplotlib.pyplot as plt
        from asr.extra_result_plotting import filrefs
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
# convex_hull
@prepare_result
class ConvexHullResult(ASRResult):
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
    ehull_description = """\
        The energy above the convex hull is the internal energy relative to the most
        stable (possibly mixed) phase of the constituent elements at T=0 K."""
    eform_description = """\
    The heat of formation (ΔH) is the internal energy of a compound relative to
    the standard states of the constituent elements at T=0 K."""

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

    @staticmethod
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
            tables[reference['title']].append(
                [name, '{:.2f} eV/atom'.format(e)])

        final_tables = []
        for title, rows in tables.items():
            final_tables.append({'type': 'table',
                                 'header': [title, ''],
                                 'rows': rows})
        return final_tables

    @staticmethod
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
                ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], '-',
                        color='lightblue')
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
    @staticmethod
    def ehull_table_rows(row, key_descriptions):
        from asr.some_row_manipulation_garbage import ehull_table_rows
        ehull = ConvexHullResult.ehull_long_description
        eform = ConvexHullResult.eform_description
        return ehull_table_rows(row, key_descriptions, ehull, eform)


# structureinfo
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


## Defects
from asr.extra_result_plotting import (
    draw_band_edge, draw_levels_occupations_labels,
)
from asr.some_row_manipulation_garbage import (
    get_spin_data,
)
from asr.randomresults import SymmetryResult, PristineResult
# defect symmetry
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

    @staticmethod
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

    @staticmethod
    def get_symmetry_tables(state_results, vbm, cbm, row, style):
        from asr.some_row_manipulation_garbage import get_symmetry_tables
        return get_symmetry_tables(state_results, vbm, cbm, row, style)

# defect info
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
    @staticmethod
    def get_concentration_row(conc_res, defect_name, q):
        from asr.some_row_manipulation_garbage import get_concentration_row
        return get_concentration_row(conc_res, defect_name, q)

# DefectLinks
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
    @staticmethod
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


## relax
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


## gs
from asr.randomresults import GapsResult, VacuumLevelResults
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

    @staticmethod
    def _explain_bandgap(row, gap_name):
        from asr.some_row_manipulation_garbage import _explain_bandgap
        return _explain_bandgap(row, gap_name)

    @staticmethod
    def vbm_or_cbm_row(title, quantity_name, reference_explanation, value):
        from asr.some_row_manipulation_garbage import vbm_or_cbm_row
        return vbm_or_cbm_row(title, quantity_name, reference_explanation,
                              value)

    @staticmethod
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


## Dynamic Stability
# phonons
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

    @staticmethod
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
                emin=np.min(energies * 1.1),
                emax=np.max([np.max(energies * 1.15),
                             0.0001]),
                ylabel='Phonon frequencies [meV]')
        plt.tight_layout()
        plt.savefig(fname)
# phonopy
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

    @staticmethod
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


## piezoelectrictensor
@prepare_result
class PiezoEleTenResult(ASRResult):
    all_voigt_labels = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
    all_voigt_indices = [[0, 1, 2, 1, 0, 0],
                         [0, 1, 2, 2, 2, 1]]

    eps_vvv: typing.List[typing.List[typing.List[float]]]
    eps_clamped_vvv: typing.List[typing.List[typing.List[float]]]

    key_descriptions = {'eps_vvv': 'Piezoelectric tensor.',
                        'eps_clamped_vvv': 'Piezoelectric tensor.'}
    formats = {"ase_webpanel": PiezoEleTenWebpanel}

    @staticmethod
    def get_voigt_indices(pbc: typing.List[bool]):
        mask = PiezoEleTenResult.get_voigt_mask(pbc)
        return [list(itertools.compress(indices, mask))
                for indices in PiezoEleTenResult.all_voigt_indices]

    @staticmethod
    def get_voigt_labels(pbc: typing.List[bool]):
        mask = PiezoEleTenResult.get_voigt_mask(pbc)
        return list(itertools.compress(PiezoEleTenResult.all_voigt_labels,
                                       mask))

    def get_voigt_mask(self, pbc_c: typing.List[bool]):
        non_pbc_axes = set(char for char, pbc in zip('xyz', pbc_c) if not pbc)

        mask = [False
                if set(voigt_label).intersection(non_pbc_axes)
                else True
                for voigt_label in self.all_voigt_labels]
        return mask


# deformationpotentials - not sure where this belongs
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
# stiffness (done)
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


# get_wfs
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
