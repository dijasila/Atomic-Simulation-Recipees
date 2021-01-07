"""DFT GW."""
from asr.core import command, option, read_json, ASRResult, prepare_result
from ase.spectrum.band_structure import BandStructure
from click import Choice
import typing
from asr.database.browser import (
    href, fig, table, describe_entry, make_panel_description)


panel_description = make_panel_description(
    """The quasiparticle (QP) band structure calculated within the G0W0
approximation from a GGA starting point. Spin-orbit interactions are included
in postprocess. The frequency dependence is treated numerically exact. For
low-dimensional materials, a truncated Coulomb interaction is used to decouple
periodic images. The QP energies are extrapolated as 1/N to the infinite plane
wave basis set limit. Spin-orbit interactions are included
non-self-consistently.""",
    articles=[
        'C2DB',
        href(
            """F. Rasmussen et al. Efficient many-body calculations for two-dimensional
            materials using exact limits for the screened potential: Band gaps
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


# This function is basically doing the exact same as HSE and could
# probably be refactored
def bs_gw(row,
          filename='gw-bs.png',
          figsize=(5.5, 5),
          fontsize=10,
          show_legend=True,
          s=0.5):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    data = row.data.get('results-asr.gw.json')
    path = data['bandstructure']['path']
    mpl.rcParams['font.size'] = fontsize
    ef = data['efermi_gw_soc']

    if row.get('evac') is not None:
        label = r'$E - E_\mathrm{vac}$ [eV]'
        reference = row.get('evac')
    else:
        label = r'$E - E_\mathrm{F}$ [eV]'
        reference = ef

    emin = row.get('vbm_gw', ef) - 3 - reference
    emax = row.get('cbm_gw', ef) + 3 - reference

    e_mk = data['bandstructure']['e_int_mk'] - reference
    x, X, labels = path.get_linear_kpoint_axis()

    # hse with soc
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
    ax.axhline(ef - reference, c='C1', ls=':')
    text = ax.annotate(
        r'$E_\mathrm{F}$',
        xy=(x0, ef - reference),
        ha='left',
        va='bottom',
        fontsize=fontsize * 1.3)
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white', alpha=0.5),
        path_effects.Normal()
    ])

    # add PBE band structure with soc
    from asr.bandstructure import add_bs_pbe
    if 'results-asr.bandstructure.json' in row.data:
        ax = add_bs_pbe(row, ax, reference=row.get('evac', row.get('efermi')),
                        color=[0.8, 0.8, 0.8])

    for Xi in X:
        ax.axvline(Xi, ls='-', c='0.5', zorder=-20)

    ax.plot([], [], **style, label='G0W0')
    plt.legend(loc='upper right')

    if not show_legend:
        ax.legend_.remove()
    plt.savefig(filename, bbox_inches='tight')


def get_kpts_size(atoms, kptdensity):
    """Try to get a reasonable monkhorst size which hits high symmetry points."""
    from gpaw.kpt_descriptor import kpts2sizeandoffsets as k2so
    size, offset = k2so(atoms=atoms, density=kptdensity)
    size[2] = 1
    for i in range(2):
        if size[i] % 6 != 0:
            size[i] = 6 * (size[i] // 6 + 1)
    kpts = {'size': size, 'gamma': True}
    return kpts


@command(requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'],
         creates=['gs_gw.gpw', 'gs_gw_nowfs.gpw'])
@option('--kptdensity', help='K-point density', type=float)
@option('--ecut', help='Plane wave cutoff', type=float)
def gs(kptdensity: float = 5.0, ecut: float = 200.0) -> ASRResult:
    """Calculate GW underlying ground state."""
    from ase.dft.bandgap import bandgap
    from gpaw import GPAW
    import numpy as np

    # check that the system is a semiconductor
    calc = GPAW('gs.gpw', txt=None)
    pbe_gap, _, _ = bandgap(calc, output=None)
    if pbe_gap < 0.05:
        raise Exception("GW: Only for semiconductors, PBE gap = "
                        + str(pbe_gap) + " eV is too small!")

    # check that the system is small enough
    atoms = calc.get_atoms()
    if len(atoms) > 4:
        raise Exception("GW: Only for small systems, "
                        + str(len(atoms)) + " > 4 atoms!")

    # setup k points/parameters
    dim = np.sum(atoms.pbc.tolist())
    if dim == 3:
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
    elif dim == 2:
        kpts = get_kpts_size(atoms=atoms, kptdensity=kptdensity)
    elif dim == 1:
        kpts = {'density': kptdensity, 'gamma': True, 'even': True}
        # TODO remove unnecessary k
        raise NotImplementedError('asr for dim=1 not implemented!')
    elif dim == 0:
        kpts = {'density': 0.0, 'gamma': True, 'even': True}
        # TODO only Gamma
        raise NotImplementedError('asr for dim=0 not implemented!')

    # we need energies/wavefunctions on the correct grid
    calc = GPAW(
        'gs.gpw',
        txt='gs_gw.txt',
        fixdensity=True,
        kpts=kpts,
        parallel={'domain': 1})
    calc.get_potential_energy()
    calc.diagonalize_full_hamiltonian(ecut=ecut)
    calc.write('gs_gw_nowfs.gpw')
    calc.write('gs_gw.gpw', mode='all')


@command(requires=['gs_gw.gpw'],
         dependencies=['asr.gw@gs'])
@option('--ecut', help='Plane wave cutoff', type=float)
@option('--mode', help='GW mode',
        type=Choice(['G0W0', 'GWG']))
def gw(ecut: float = 200.0, mode: str = 'G0W0') -> ASRResult:
    """Calculate GW corrections."""
    from ase.dft.bandgap import bandgap
    from gpaw import GPAW
    from gpaw.response.g0w0 import G0W0
    import numpy as np

    # check that the system is a semiconductor
    calc = GPAW('gs.gpw', txt=None)
    pbe_gap, _, _ = bandgap(calc, output=None)
    if pbe_gap < 0.05:
        raise Exception("GW: Only for semiconductors, PBE gap = "
                        + str(pbe_gap) + " eV is too small!")

    # check that the system is small enough
    atoms = calc.get_atoms()
    if len(atoms) > 4:
        raise Exception("GW: Only for small systems, "
                        + str(len(atoms)) + " > 4 atoms!")

    # Setup parameters
    dim = np.sum(atoms.pbc.tolist())
    if dim == 3:
        truncation = 'wigner-seitz'
        q0_correction = False
    elif dim == 2:
        truncation = '2D'
        q0_correction = True
    elif dim == 1:
        raise NotImplementedError('asr for dim=1 not implemented!')
        truncation = '1D'
        q0_correction = False
    elif dim == 0:
        raise NotImplementedError('asr for dim=0 not implemented!')
        truncation = '0D'
        q0_correction = False

    if mode == 'GWG':
        raise NotImplementedError('GW: asr for GWG not implemented!')

    lb, ub = max(calc.wfs.nvalence // 2 - 8, 0), calc.wfs.nvalence // 2 + 4

    calc = G0W0(calc='gs_gw.gpw',
                bands=(lb, ub),
                ecut=ecut,
                ecut_extrapolation=True,
                truncation=truncation,
                nblocksmax=True,
                q0_correction=q0_correction,
                filename='g0w0',
                restartfile='g0w0.tmp',
                savepckl=False)

    results = calc.calculate()
    results['minband'] = lb
    results['maxband'] = ub
    return results


@command(requires=['results-asr.gw@gw.json'],
         dependencies=['asr.gw@gw'])
@option('-c', '--correctgw', is_flag=True, default=False)
@option('-z', '--empz', type=float, default=0.75,
        help='Replacement Z for unphysical Zs')
def empirical_mean_z(correctgw: bool = True,
                     empz: float = 0.75) -> ASRResult:
    """Apply the empirical-Z method.

    Implements the method described in https://arxiv.org/abs/2009.00314.

    This method consists of replacing the G0W0 Z-value with the empirical
    mean of Z-values (calculated from C2DB GW calculations) whenever the
    G0W0 is "quasiparticle-inconsistent", i.e. the G0W0 Z is outside the
    interval [0.5, 1.0]. The empirical mean Z was found to be

    Z0 = 0.75.

    Pseudocode:

    For all states:
        if Z not in [0.5, 1.0]:
            set GW energy = E_KS + Z0 * (Sigma_GW - vxc + exx)

    The last line can be implemented as

    new GW energy = E_KS + (Old GW - E_KS) * Z0 / Z
    """
    import numpy as np
    gwresults = read_json('results-asr.gw@gw.json')
    if not correctgw:
        return gwresults

    Z0 = empz
    results = gwresults.copy()

    Z_skn = gwresults['Z']
    e_skn = gwresults['eps']
    qp_skn = gwresults['qp']
    results['qpGW'] = qp_skn.copy()

    indices = np.logical_not(np.logical_and(Z_skn >= 0.5, Z_skn <= 1.0))
    qp_skn[indices] = e_skn[indices] + \
        (qp_skn[indices] - e_skn[indices]) * Z0 / Z_skn[indices]

    results['qp'] = qp_skn

    return results


def webpanel(result, row, key_descriptions):

    prop = table(row, 'Property', [
        'gap_gw', 'gap_dir_gw',
    ], key_descriptions)

    if row.get('evac'):
        prop['rows'].extend(
            [['Valence band maximum wrt. vacuum level (G0W0)',
              f'{row.vbm_gw - row.evac:.2f} eV'],
             ['Conduction band minimum wrt. vacuum level (G0W0)',
              f'{row.cbm_gw - row.evac:.2f} eV']])
    else:
        prop['rows'].extend(
            [['Valence band maximum wrt. Fermi level (G0W0)',
              f'{row.vbm_gw - row.efermi:.2f} eV'],
             ['Conduction band minimum wrt. Fermi level (G0W0)',
              f'{row.cbm_gw - row.efermi:.2f} eV']])

    panel = {'title': describe_entry('Electronic band structure (G0W0)',
                                     panel_description),
             'columns': [[fig('gw-bs.png')], [fig('bz-with-gaps.png'), prop]],
             'plot_descriptions': [{'function': bs_gw,
                                    'filenames': ['gw-bs.png']}],
             'sort': 16}

    if row.get('gap_gw'):
        description = (
            'The electronic band gap calculated with '
            'G0W0 including spin-orbit effects. \n\n'
        )
        rows = [
            [
                describe_entry('Band gap (G0W0)', description),
                f'{row.gap_gw:0.2f} eV'
            ]
        ]

        summary = {'title': 'Summary',
                   'columns': [[{'type': 'table',
                                 'header': ['Electronic properties', ''],
                                 'rows': rows}]],
                   'sort': 12}

        return [panel, summary]

    return [panel]


@prepare_result
class Result(ASRResult):

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
        "vbm_gw_nosoc": "Valence band maximum w/o soc. (G0W0) [eV]",
        "cbm_gw_nosoc": "Conduction band minimum w/o soc. (G0W0) [eV]",
        "gap_dir_gw_nosoc": "Direct gap w/o soc. (G0W0) [eV]",
        "gap_gw_nosoc": "Gap w/o soc. (G0W0) [eV]",
        "kvbm_nosoc": "k-point of G0W0 valence band maximum w/o soc",
        "kcbm_nosoc": "k-point of G0W0 conduction band minimum w/o soc",
        "vbm_gw": "Valence band maximum (G0W0) [eV]",
        "cbm_gw": "Conduction band minimum (G0W0) [eV]",
        "gap_dir_gw": "Direct band gap (G0W0) [eV]",
        "gap_gw": "Band gap (G0W0) [eV]",
        "kvbm": "k-point of G0W0 valence band maximum",
        "kcbm": "k-point of G0W0 conduction band minimum",
        "efermi_gw_nosoc": "Fermi level w/o soc. (G0W0) [eV]",
        "efermi_gw_soc": "Fermi level (G0W0) [eV]",
        "bandstructure": "GW bandstructure."
    }
    formats = {"ase_webpanel": webpanel}


@command(requires=['gs_gw_nowfs.gpw',
                   'results-asr.gw@empirical_mean_z.json',
                   'results-asr.bandstructure.json'],
         dependencies=['asr.bandstructure', 'asr.gw@empirical_mean_z'],
         returns=Result)
def main() -> Result:
    import numpy as np
    from gpaw import GPAW
    from asr.utils import fermi_level
    from ase.dft.bandgap import bandgap
    from asr.hse import MP_interpolate
    from types import SimpleNamespace

    calc = GPAW('gs_gw_nowfs.gpw', txt=None)
    gwresults = SimpleNamespace(**read_json('results-asr.gw@empirical_mean_z.json'))

    lb = gwresults.minband
    ub = gwresults.maxband

    delta_skn = gwresults.qp - gwresults.eps

    # Interpolate band structure
    results = MP_interpolate(calc, delta_skn, lb, ub)

    # First get stuff without SOC
    eps_skn = gwresults.qp
    efermi_nosoc = fermi_level(calc, eigenvalues=eps_skn,
                               nelectrons=(calc.get_number_of_electrons()
                                           - 2 * lb),
                               nspins=eps_skn.shape[0])
    gap, p1, p2 = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps_skn, efermi=efermi_nosoc,
                             direct=True, output=None)
    if gap > 0:
        ibzkpts = calc.get_ibz_k_points()
        kvbm_nosoc = ibzkpts[p1[1]]  # k coordinates of vbm
        kcbm_nosoc = ibzkpts[p2[1]]  # k coordinates of cbm
        vbm = eps_skn[p1]
        cbm = eps_skn[p2]
        subresults = {'vbm_gw_nosoc': vbm,
                      'cbm_gw_nosoc': cbm,
                      'gap_dir_gw_nosoc': gapd,
                      'gap_gw_nosoc': gap,
                      'kvbm_nosoc': kvbm_nosoc,
                      'kcbm_nosoc': kcbm_nosoc}
    else:
        subresults = {'vbm_gw_nosoc': None,
                      'cbm_gw_nosoc': None,
                      'gap_dir_gw_nosoc': None,
                      'gap_gw_nosoc': None,
                      'kvbm_nosoc': None,
                      'kcbm_nosoc': None}
    results.update(subresults)

    # Get the SO corrected GW QP energires
    from gpaw.spinorbit import soc_eigenstates
    from asr.magnetic_anisotropy import get_spin_axis
    theta, phi = get_spin_axis()
    soc = soc_eigenstates(calc, eigenvalues=eps_skn,
                          n1=lb, n2=ub,
                          theta=theta, phi=phi)

    eps_skn = soc.eigenvalues()[np.newaxis]  # e_skm, dummy spin index
    efermi_soc = fermi_level(calc, eigenvalues=eps_skn,
                             nelectrons=(calc.get_number_of_electrons()
                                         - 2 * lb),
                             nspins=2)
    gap, p1, p2 = bandgap(eigenvalues=eps_skn, efermi=efermi_soc,
                          output=None)
    gapd, p1d, p2d = bandgap(eigenvalues=eps_skn, efermi=efermi_soc,
                             direct=True, output=None)
    if gap:
        bzkpts = calc.get_bz_k_points()
        kvbm = bzkpts[p1[1]]
        kcbm = bzkpts[p2[1]]
        vbm = eps_skn[p1]
        cbm = eps_skn[p2]
        subresults = {'vbm_gw': vbm,
                      'cbm_gw': cbm,
                      'gap_dir_gw': gapd,
                      'gap_gw': gap,
                      'kvbm': kvbm,
                      'kcbm': kcbm}
    else:
        subresults = {'vbm_gw': None,
                      'cbm_gw': None,
                      'gap_dir_gw': None,
                      'gap_gw': None,
                      'kvbm': None,
                      'kcbm': None}
    results.update(subresults)
    results.update({'efermi_gw_nosoc': efermi_nosoc,
                    'efermi_gw_soc': efermi_soc})

    return Result(data=results)


if __name__ == '__main__':
    main.cli()
