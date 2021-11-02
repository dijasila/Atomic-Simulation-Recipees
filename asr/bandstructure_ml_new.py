"""Electronic band structures using machine learning model."""
from typing import Union
import numpy as np
from ase.dft.kpoints import labels_from_kpts
from ase.parallel import world,parprint
from asr.core import command, option, ASRResult, singleprec_dict, prepare_result, read_json
from asr.database.browser import fig, make_panel_description, describe_entry
from asr.utils.hacks import gs_xcname_from_row
import os


panel_description = make_panel_description(
    """Machine learning band structures"""
)

@command('asr.bandstructure_ml_new',
         requires=['gs.gpw','bs.gpw'],
         creates=['bs_matrix_elements.npz','bs_ml.gpw'])
def calculate():
    """Calculate electronic band structure using machine learning."""
    import sys
    sys.path.append('/home/niflheim/nirkn/electronic-structure-machine-learning/')
    from efp import MLGPAW
    ML = MLGPAW('bs.gpw')
    ML.update_eigenvalues(interpolate=10)

bs_ml_png = 'bs_ml.png'
#bs_html = 'bs.html'



def add_bs_ks(row, ax, reference=0, color='C1'):
    """Plot with soc on ax."""
    d = row.data.get('results-asr.bandstructure_ml_new.json')
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


def legend_on_top(ax, **kwargs):
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1, 1, 0),
              mode='expand', **kwargs)


def plot_bs(row,
            filename
            ):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    data = row.data['results-asr.bandstructure_ml_new.json']
    bs_label = 'ML-G₀W₀@PBE'
    efermi = data['bs_ml_soc']['efermi']
    vbm = None
    cbm = None

    figsize = (5.5, 5)
    fontsize = 10

    path = data['bs_ml_soc']['path']

    reference = row.get('evac')
    if reference is None:
        reference = efermi
        label = r'$E - E_\mathrm{F}$ [eV]'
    else:
        label = r'$E - E_\mathrm{vac}$ [eV]'

    emin_offset = efermi if vbm is None else vbm
    emax_offset = efermi if cbm is None else cbm
    emin = emin_offset - 7.5 - reference
    emax = emax_offset + 7.5 - reference

    e_mk = data['bs_ml_soc']['energies'] - reference
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
    ax = add_bs_ks(row, ax, reference=row.get('evac', row.get('efermi')),
                    color=[0.8, 0.8, 0.8])

    for Xi in X:
        ax.axvline(Xi, ls='-', c='0.5', zorder=-20)

    ax.plot([], [], **style, label=bs_label)
    legend_on_top(ax, ncol=2)
    plt.savefig(filename, bbox_inches='tight')


def webpanel(result, row, key_descriptions):
    from typing import Tuple, List
    from asr.utils.hacks import gs_xcname_from_row


    panel = {'title': describe_entry(f'Electronic band structure (ML)',
                                     panel_description),
             'columns': [
                 [
                     fig(bs_ml_png),
                 ],
                 [fig('bz-with-gaps.png')]],
             'plot_descriptions': [{'function': plot_bs,
                                    'filenames': [bs_ml_png]}],
             'sort': 12}

    return [panel]


@prepare_result
class Result(ASRResult):

    version: int = 0

    bs_soc: dict
    bs_nosoc: dict
    bs_ml_soc: dict
    bs_ml_nosoc: dict

    key_descriptions = \
        {
            'bs_soc': 'Bandstructure data with spin–orbit coupling.',
            'bs_nosoc': 'Bandstructure data without spin–orbit coupling.',
            'bs_ml_soc': 'ML bandstructure data with spin–orbit coupling.',
            'bs_ml_nosoc': 'ML bandstructure data without spin–orbit coupling.'
        }

    formats = {"ase_webpanel": webpanel}


@command('asr.bandstructure_ml_new',
         #requires=['gs.gpw', 'bs.gpw','bs_ml.gpw', 'results-asr.gs.json'],
         dependencies=['asr.bandstructure_ml_new@calculate'],
         returns=Result)
def main() -> Result:
    from gpaw import GPAW
    from ase.spectrum.band_structure import get_band_structure
    from ase.dft.kpoints import BandPath
    from ase.spectrum.band_structure import BandStructure
    from asr.core import read_json
    import copy
    from asr.utils.gpw2eigs import gpw2eigs
    from asr.magnetic_anisotropy import get_spin_axis, get_spin_index
    import xgboost as xgb

    # PBE
    gsresults = read_json('results-asr.gs.json')
    ref = gsresults['efermi']
    calc = GPAW('bs.gpw', txt=None)
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
    efermi_nosoc = gsresults['gaps_nosoc']['efermi']
    bsresults['efermi'] = efermi_nosoc

    # We copy the bsresults dict because next we will add SOC
    results['bs_nosoc'] = copy.deepcopy(bsresults)  # BS with no SOC

    # Add spin orbit correction
    bsresults = bs.todict()

    theta, phi = get_spin_axis()

    # We use a larger symmetry tolerance because we want to correctly
    # color spins which doesn't always happen due to slightly broken
    # symmetries, hence tolerance=1e-2.
    e_km, _, s_kvm = gpw2eigs(
        'bs.gpw', soc=True, return_spin=True, theta=theta, phi=phi,
        symmetry_tolerance=1e-2)
    bsresults['energies'] = e_km.T
    efermi = gsresults['efermi']
    bsresults['efermi'] = efermi

    # Get spin projections for coloring of bandstructure
    path = bsresults['path']
    npoints = len(path.kpts)
    s_mvk = np.array(s_kvm.transpose(2, 1, 0))

    if s_mvk.ndim == 3:
        sz_mk = s_mvk[:, get_spin_index(), :]  # take x, y or z component
    else:
        sz_mk = s_mvk

    assert sz_mk.shape[1] == npoints, f'sz_mk has wrong dims, {npoints}'

    bsresults['sz_mk'] = sz_mk

    # ML
    gsresults = read_json('results-asr.gs_ml.json')
    ref = gsresults['efermi']
    calc = GPAW('bs_ml.gpw', txt=None)
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
    bs_ml = get_band_structure(calc=calc, path=path, reference=ref)

    results_ml = {}
    bsresults_ml = bs_ml.todict()

    # Save Fermi levels
    efermi_nosoc = gsresults['gaps_nosoc']['efermi']
    bsresults_ml['efermi'] = efermi_nosoc

    # We copy the bsresults dict because next we will add SOC
    results_ml['bs_nosoc'] = copy.deepcopy(bsresults_ml)  # BS with no SOC

    # Add spin orbit correction
    bsresults_ml = bs_ml.todict()

    theta, phi = get_spin_axis()

    # We use a larger symmetry tolerance because we want to correctly
    # color spins which doesn't always happen due to slightly broken
    # symmetries, hence tolerance=1e-2.
    e_km, _, s_kvm = gpw2eigs(
        'bs_ml.gpw', soc=True, return_spin=True, theta=theta, phi=phi,
        symmetry_tolerance=1e-2)
    bsresults_ml['energies'] = e_km.T
    efermi = gsresults['efermi']
    bsresults_ml['efermi'] = efermi

    # Get spin projections for coloring of bandstructure
    path = bsresults_ml['path']
    npoints = len(path.kpts)
    s_mvk = np.array(s_kvm.transpose(2, 1, 0))

    if s_mvk.ndim == 3:
        sz_mk = s_mvk[:, get_spin_index(), :]  # take x, y or z component
    else:
        sz_mk = s_mvk

    assert sz_mk.shape[1] == npoints, f'sz_mk has wrong dims, {npoints}'

    bsresults_ml['sz_mk'] = sz_mk

    return Result.fromdata(
        bs_soc=singleprec_dict(bsresults),
        bs_nosoc=singleprec_dict(results['bs_nosoc']),
        bs_ml_soc=singleprec_dict(bsresults_ml),
        bs_ml_nosoc=singleprec_dict(results_ml['bs_nosoc'])
    )


if __name__ == '__main__':
    main.cli()
