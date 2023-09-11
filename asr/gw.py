"""DFT GW."""
from asr.core import command, option, read_json, ASRResult, prepare_result
from ase.spectrum.band_structure import BandStructure
from click import Choice
import typing
from asr.webpages.browser import make_panel_description
from asr.webpages.appresources import HTMLStringFormat
from asr.utils.gw_hse import GWHSEInfo
from asr.utils.kpts import get_kpts_size


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
            HTMLStringFormat.href(
                """F. Rasmussen et al. Efficient many-body calculations for
two-dimensional materials using exact limits for the screened potential: Band gaps
of MoS2, h-BN, and phosphorene, Phys. Rev. B 94, 155406 (2016)""",
                'https://doi.org/10.1103/PhysRevB.94.155406',
            ),
            HTMLStringFormat.href(
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
        from asr.hse import plot_bs
        data = row.data['results-asr.gw.json']
        return plot_bs(row, filename=filename, bs_label='G₀W₀',
                       data=data,
                       efermi=data['efermi_gw_soc'],
                       cbm=row.get('cbm_gw'),
                       vbm=row.get('vbm_gw'))


@command(requires=['gs.gpw'],
         dependencies=['asr.gs@calculate'],
         creates=['gs_gw.gpw', 'gs_gw_nowfs.gpw'])
@option('--kptdensity', help='K-point density', type=float)
@option('--ecut', help='Plane wave cutoff', type=float)
def gs(kptdensity: float = 5.0, ecut: float = 200.0) -> ASRResult:
    """Calculate GW underlying ground state."""
    from ase.dft.bandgap import bandgap
    from gpaw import GPAW

    # check that the system is a semiconductor
    calc = GPAW('gs.gpw', txt=None)
    scf_gap, _, _ = bandgap(calc, output=None)
    if scf_gap < 0.05:
        raise Exception("GW: Only for semiconductors, SCF gap = "
                        + str(scf_gap) + " eV is too small!")

    # check that the system is small enough
    atoms = calc.get_atoms()
    if len(atoms) > 4:
        raise Exception("GW: Only for small systems, "
                        + str(len(atoms)) + " > 4 atoms!")

    # setup k points/parameters
    dim = sum(atoms.pbc)
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
    calc = GPAW('gs.gpw').fixed_density(
        txt='gs_gw.txt',
        kpts=kpts,
        parallel={'domain': 1})
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

    # check that the system is a semiconductor
    calc = GPAW('gs.gpw', txt=None)
    scf_gap, _, _ = bandgap(calc, output=None)
    if scf_gap < 0.05:
        raise Exception("GW: Only for semiconductors, SCF gap = "
                        + str(scf_gap) + " eV is too small!")

    # check that the system is small enough
    atoms = calc.get_atoms()
    if len(atoms) > 4:
        raise Exception("GW: Only for small systems, "
                        + str(len(atoms)) + " > 4 atoms!")

    # Setup parameters
    dim = sum(atoms.pbc)
    if dim == 3:
        truncation = 'wigner-seitz'
        q0_correction = False
    elif dim == 2:
        truncation = '2D'
        q0_correction = True
    else:
        raise NotImplementedError(f'dim={dim} not implemented!')

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
    from asr.utils.gw_hse import gw_hse_webpanel
    return gw_hse_webpanel(result, row, key_descriptions, GWInfo(row),
                           sort=16)


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
