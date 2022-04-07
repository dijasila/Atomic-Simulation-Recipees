"""HSE06 band structure."""
from asr.calculators import Calculation
import asr
from asr.core import command, ASRResult, prepare_result
import typing
from ase.spectrum.band_structure import BandStructure
from asr.c2db.bandstructure import legend_on_top
from asr.database.browser import make_panel_description
from asr.utils.gw_hse import GWHSEInfo
from asr.utils.kpts import get_kpts_size


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
    def plot_bs(context, filename):
        results = context.result
        return plot_bs(context, filename=filename, bs_label='HSE06',
                       data=results,
                       efermi=results['efermi_hse_soc'],
                       vbm=results['vbm_hse'],
                       cbm=results['cbm_hse'])


@prepare_result
class HSECalculationResult(ASRResult):

    hse_eigenvalues: typing.List[float]
    hse_eigenvalues_soc: typing.List[float]
    calculation: Calculation

    key_descriptions = dict(
        calculation='Calculation object',
        hse_eigenvalues='HSE eigenvalues without SOC.',
        hse_eigenvalues_soc='HSE eigenvalues with SOC.',
    )


@command(module='asr.c2db.hse')
#@option('--kptdensity', help='K-point density', type=float)
#@option('--emptybands', help='number of empty bands to include', type=int)
def calculate(
        gsresult,
        mag_ani,
        kptdensity: float = 8.0,
        emptybands: int = 20,
) -> HSECalculationResult:
    """Calculate HSE06 corrections."""
    eigs, calc, hse_nowfs = hse(
        kptdensity=kptdensity,
        emptybands=emptybands,
        gsresult=gsresult,
    )
    eigs_soc = hse_spinorbit(e_skn=eigs.get('e_skn'), calc=calc, mag_ani=mag_ani)
    results = {
        'hse_eigenvalues': eigs,
        'hse_eigenvalues_soc': eigs_soc,
        'calculation': hse_nowfs,
    }
    return HSECalculationResult(data=results)


def hse(kptdensity, emptybands, gsresult):
    from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues

    convbands = int(emptybands / 2)
    calc = gsresult.calculation.load(parallel={'band': 1, 'kpt': 1})
    atoms = calc.atoms

    ND = sum(atoms.pbc)
    if ND == 3 or ND == 1:
        kpts = {'density': kptdensity, 'gamma': True, 'even': False}
    elif ND == 2:
        kpts = get_kpts_size(atoms=atoms, kptdensity=kptdensity)

    calc.set(nbands=-emptybands,
             fixdensity=True,
             kpts=kpts,
             convergence={'bands': -convbands},
             txt='hse.txt')
    calc.get_potential_energy()
    hse_nowfs = calc.save(id='hse_nowfs')
    nb = calc.get_number_of_bands()
    # XXX gpaw does not like the GPAWLikeAdaptor which we have.
    result = non_self_consistent_eigenvalues(calc.calculator,
                                             'HSE06',
                                             n1=0,
                                             n2=nb - convbands,
                                             snapshot='hse-snapshot.json')
    e_scf_skn, vxc_scf_skn, vxc_hse_skn = result
    e_hse_skn = e_scf_skn - vxc_scf_skn + vxc_hse_skn

    dct = dict(vxc_hse_skn=vxc_hse_skn,
               e_scf_skn=e_scf_skn,
               vxc_scf_skn=vxc_scf_skn,
               e_hse_skn=e_hse_skn)
    return dct, calc, hse_nowfs


def hse_spinorbit(e_skn, calc, mag_ani):
    from gpaw.spinorbit import soc_eigenstates

    dct_soc = {}
    theta, phi = mag_ani.spin_angles()

    soc = soc_eigenstates(calc,
                          eigenvalues=e_skn,
                          theta=theta, phi=phi)
    dct_soc['e_hse_mk'] = soc.eigenvalues().T
    dct_soc['s_hse_mk'] = soc.spin_projections()[
        :, :,
        mag_ani.spin_index()].T
    return dct_soc


def MP_interpolate(
        results_bandstructure,
        bscalculateres,
        calc,
        delta_skn,
        lb,
        ub,
        mag_ani,
):
    """Interpolate corrections to band patch.

    Calculates band stucture along the same band path used for SCF
    by interpolating a correction onto the SCF band structure.
    """
    import numpy as np
    from gpaw.spinorbit import soc_eigenstates
    from ase.dft.kpoints import (get_monkhorst_pack_size_and_offset,
                                 monkhorst_pack_interpolate)
    from asr.core import singleprec_dict

    bandrange = np.arange(lb, ub)

    # SCF (without SOC)
    path = results_bandstructure['bs_nosoc']['path']
    e_scf_skn = results_bandstructure['bs_nosoc']['energies']

    size, offset = get_monkhorst_pack_size_and_offset(calc.get_bz_k_points())
    bz2ibz = calc.get_bz_to_ibz_map()
    icell = calc.atoms.cell.reciprocal()
    eps = monkhorst_pack_interpolate(path.kpts, delta_skn.transpose(1, 0, 2),
                                     icell, bz2ibz, size, offset)
    delta_interp_skn = eps.transpose(1, 0, 2)
    e_int_skn = e_scf_skn[:, :, bandrange] + delta_interp_skn
    dct = dict(e_int_skn=e_int_skn, path=path)

    # add SOC from bs.gpw
    calc = bscalculateres.calculation.load()
    theta, phi = mag_ani.spin_angles()
    soc = soc_eigenstates(calc, eigenvalues=e_int_skn,
                          n1=lb, n2=ub,
                          theta=theta, phi=phi)
    dct.update(e_int_mk=soc.eigenvalues().T)

    results = {}
    results['bandstructure'] = singleprec_dict(dct)

    return results


def plot_bs(context,
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

    eref = context.energy_reference()
    reference = eref.value

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
    ax.set_ylabel(eref.mpl_plotlabel())
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
    # XXXXX HSE does not depend on bandstruture, so combining those
    # in the same plot is not the problem of the HSE recipe!
    #
    # from asr.c2db.bandstructure import add_bs_ks
    # if 'results-asr.c2db.bandstructure.json' in row.data:
    #     ax = add_bs_ks(context, ax, reference=eref.value,
    #                    color=[0.8, 0.8, 0.8])

    for Xi in X:
        ax.axvline(Xi, ls='-', c='0.5', zorder=-20)

    ax.plot([], [], **style, label=bs_label)
    legend_on_top(ax, ncol=2)
    plt.savefig(filename, bbox_inches='tight')


def webpanel(result, context):
    from asr.utils.gw_hse import gw_hse_webpanel
    return gw_hse_webpanel(result, context, HSEInfo(result), sort=15)


@prepare_result
class Result(ASRResult):
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
    formats = {"webpanel2": webpanel}


@command(module='asr.c2db.hse')
#@option('--kptdensity', help='K-point density', type=float)
#@option('--emptybands', help='number of empty bands to include', type=int)
#@asr.calcopt(
#    aliases=['-b', '--bsrestart'],
#    help='Bandstructure Calculator params.',
#    matcher=asr.matchers.EQUAL,
    #)
#@option('--kptpath', type=str, help='Custom kpoint path.')
#@option('--npoints',
#        type=int,
#        help='Number of points along k-point path.')
def postprocess(
        results_hse,
        results_bs_post,
        results_bs_calculate,
        mag_ani,
) -> Result:
    """Interpolate HSE band structure along a given path."""
    import numpy as np
    from asr.utils import fermi_level
    from ase.dft.bandgap import bandgap

    calc = results_hse.calculation.load()
    data = results_hse['hse_eigenvalues']
    nbands = data['e_hse_skn'].shape[2]
    # Key used to be caled vxc_pbe_skn but is now called vxc_scf_skn.
    # We allow the old name for compatibility:
    vxc_scf_skn = data.get('vxc_scf_skn', data.get('vxc_pbe_skn'))

    delta_skn = data['vxc_hse_skn'] - vxc_scf_skn
    results = MP_interpolate(
        results_bandstructure=results_bs_post,
        bscalculateres=results_bs_calculate,
        calc=calc,
        delta_skn=delta_skn,
        lb=0,
        ub=nbands,
        mag_ani=mag_ani)

    # get gap, cbm, vbm, etc...
    eps_skn = results_hse['hse_eigenvalues']['e_hse_skn']
    ibzkpts = calc.get_ibz_k_points()
    efermi_nosoc = fermi_level(calc, eigenvalues=eps_skn,
                               nspins=eps_skn.shape[0])
    gap, p1, p2 = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                             direct=True, output=None)
    if gap:
        kvbm_nosoc = ibzkpts[p1[1]]  # k coordinates of vbm
        kcbm_nosoc = ibzkpts[p2[1]]  # k coordinates of cbm
        vbm = eps_skn[p1]
        cbm = eps_skn[p2]
        subresults = {'vbm_hse_nosoc': vbm,
                      'cbm_hse_nosoc': cbm,
                      'gap_dir_hse_nosoc': gapd,
                      'gap_hse_nosoc': gap,
                      'kvbm_nosoc': kvbm_nosoc,
                      'kcbm_nosoc': kcbm_nosoc}
    else:
        subresults = {'vbm_hse_nosoc': None,
                      'cbm_hse_nosoc': None,
                      'gap_dir_hse_nosoc': gapd,
                      'gap_hse_nosoc': gap,
                      'kvbm_nosoc': None,
                      'kcbm_nosoc': None}
    results.update(subresults)

    eps = results_hse['hse_eigenvalues_soc']['e_hse_mk']
    eps = eps.transpose()[np.newaxis]  # e_skm, dummy spin index
    efermi_soc = fermi_level(calc, eigenvalues=eps, nspins=2)
    bzkpts = calc.get_bz_k_points()
    gap, p1, p2 = bandgap(eigenvalues=eps, efermi=efermi_soc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps, efermi=efermi_soc,
                             direct=True, output=None)
    if gap:
        kvbm = bzkpts[p1[1]]
        kcbm = bzkpts[p2[1]]
        vbm = eps[p1]
        cbm = eps[p2]
        subresults = {'vbm_hse': vbm,
                      'cbm_hse': cbm,
                      'gap_dir_hse': gapd,
                      'gap_hse': gap,
                      'kvbm': kvbm,
                      'kcbm': kcbm}
    else:
        subresults = {'vbm_hse': None,
                      'cbm_hse': None,
                      'gap_dir_hse': gapd,
                      'gap_hse': gap,
                      'kvbm': None,
                      'kcbm': None}
    results.update(subresults)

    subresults = {'efermi_hse_nosoc': efermi_nosoc,
                  'efermi_hse_soc': efermi_soc}
    results.update(subresults)

    return Result(data=results)


sel = asr.Selector()
sel.name = sel.EQ("asr.c2db.hse:main")
sel.version = sel.EQ(-1)


@asr.mutation(selector=sel)
def add_missing_hse_keys(record: asr.Record) -> asr.Record:
    """Add missing keys in old HSE results."""
    res = record.result
    if isinstance(res, ASRResult):
        data = res.data
    else:
        assert isinstance(res, dict)
        data = res

    keys_and_values = [
        "vbm_hse_nosoc", None,
        "cbm_hse_nosoc", None,
        "gap_dir_hse_nosoc", 0,
        "gap_hse_nosoc", 0,
        "kvbm_nosoc", None,
        "kcbm_nosoc", None,
        "vbm_hse", None,
        "cbm_hse", None,
        "gap_dir_hse", 0,
        "gap_hse", 0,
        "kvbm", None,
        "kcbm", None,
    ]

    for key, default_value in zip(keys_and_values[::2], keys_and_values[1::2]):
        if key not in data:
            data[key] = default_value
    new_result = Result.fromdata(**data)
    record.result = new_result
    return record


class HSEWorkflow:
    def __init__(self, rn, bsworkflow):
        gsw = bsworkflow.gsworkflow
        self.bsworkflow = bsworkflow

        mag_ani = gsw.magnetic_anisotropy.output

        self.calculate = rn.task(
            'asr.c2db.hse.calculate',
            gsresult=gsw.scf.output,
            mag_ani=mag_ani)

        self.postprocess = rn.task(
            'asr.c2db.hse.postprocess',
            results_hse=self.calculate.output,
            results_bs_post=bsworkflow.postprocess.output,
            results_bs_calculate=bsworkflow.bs.output,
            mag_ani=mag_ani)
