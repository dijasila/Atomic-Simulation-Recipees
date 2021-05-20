"""DFT GW."""
import numpy as np
from ase import Atoms
import asr
from asr.core import (
    command, option, ASRResult, prepare_result,
    atomsopt, calcopt, ExternalFile,
)
from asr.gs import calculate as calculategs
from asr.bandstructure import main as bsmain
from ase.spectrum.band_structure import BandStructure
from click import Choice
import typing
from asr.database.browser import href, make_panel_description
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
        from asr.hse import plot_bs
        data = row.data['results-asr.gw.json']
        return plot_bs(row, filename=filename, bs_label='G₀W₀',
                       data=data,
                       efermi=data['efermi_gw_soc'],
                       cbm=row.get('cbm_gw'),
                       vbm=row.get('vbm_gw'))


@command()
@atomsopt
@calcopt
@option('--kptdensity', help='K-point density', type=float)
@option('--ecut', help='Plane wave cutoff', type=float)
def gs(
        atoms: Atoms,
        calculator: dict = calculategs.defaults.calculator,
        kptdensity: float = 5.0,
        ecut: float = 200.0,
) -> ASRResult:
    """Calculate GW underlying ground state."""
    from ase.dft.bandgap import bandgap
    # check that the system is a semiconductor
    res = calculategs(atoms=atoms, calculator=calculator)
    calc = res.calculation.load()
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
    calc = res.calculation.load()
    calc = calc.fixed_density(
        txt='gs_gw.txt',
        kpts=kpts,
        parallel={'domain': 1})
    calc.diagonalize_full_hamiltonian(ecut=ecut)
    gs_gw_nowfs = 'gs_gw_nowfs.gpw'
    gs_gw = 'gs_gw.gpw'
    calc.write(gs_gw_nowfs)
    calc.write(gs_gw, mode='all')

    return {
        gs_gw_nowfs: ExternalFile.fromstr(gs_gw_nowfs),
        gs_gw: ExternalFile.fromstr(gs_gw),
    }


@command()
@atomsopt
@calcopt
@option('--kptdensity', help='K-point density', type=float)
@option('--ecut', help='Plane wave cutoff', type=float)
@option('--mode', help='GW mode',
        type=Choice(['G0W0', 'GWG']))
def gw(atoms: Atoms,
       calculator: dict = gs.defaults.calculator,
       kptdensity: float = gs.defaults.kptdensity,
       ecut: float = gs.defaults.ecut,
       mode: str = 'G0W0') -> dict:
    """Calculate GW corrections."""
    from ase.dft.bandgap import bandgap
    from gpaw.response.g0w0 import G0W0

    # check that the system is a semiconductor
    res = calculategs(atoms=atoms, calculator=calculator)
    calc = res.calculation.load()
    scf_gap, _, _ = bandgap(calc, output=None)

    if len(atoms) > 4:
        raise Exception("GW: Only for small systems, "
                        + str(len(atoms)) + " > 4 atoms!")

    if scf_gap < 0.05:
        raise Exception("GW: Only for semiconductors, SCF gap = "
                        + str(scf_gap) + " eV is too small!")

    res = gs(
        atoms=atoms,
        calculator=calculator,
        kptdensity=kptdensity,
        ecut=ecut,
    )

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

    calc = G0W0(calc=res['gs_gw.gpw'],
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


@command()
@atomsopt
@calcopt
@option('--kptdensity', help='K-point density', type=float)
@option('--ecut', help='Plane wave cutoff', type=float)
@option('--mode', help='GW mode',
        type=Choice(['G0W0', 'GWG']))
@option('-c', '--correctgw', is_flag=True, default=False)
@option('-z', '--empz', type=float, default=0.75,
        help='Replacement Z for unphysical Zs')
def empirical_mean_z(
        atoms: Atoms,
        calculator: dict = gw.defaults.calculator,
        kptdensity: float = gw.defaults.kptdensity,
        ecut: float = gw.defaults.ecut,
        mode: str = gw.defaults.mode,
        correctgw: bool = True,
        empz: float = 0.75,
) -> dict:
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
    gwresults = gw(
        atoms=atoms,
        calculator=calculator,
        kptdensity=kptdensity,
        ecut=ecut,
        mode=mode,
    )
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


sel = asr.Selector()
sel.version = sel.EQ(-1)
sel.name = sel.EQ('asr.gw:main')


@asr.migration(selector=sel)
def migrate_1(record):
    """Prepare record for resultfile migration."""
    emptybands = (
        record.parameters.dependency_parameters[
            'asr.bandstructure:calculate']['emptybands']
    )
    record.parameters.ecut = 200.0
    record.parameters.kptdensity = 5.0
    record.parameters.mode = 'G0W0'
    record.parameters.bsrestart = {
        'nbands': -emptybands,
        'txt': 'bs.txt',
        'fixdensity': True,
        'convergence': {
            'bands': -emptybands // 2},
        'symmetry': 'off'
    }
    del record.parameters.dependency_parameters[
        'asr.bandstructure:calculate']['emptybands']
    return record


@command(
    migrations=[migrate_1]
)
@atomsopt
@calcopt
@asr.calcopt(
    aliases=['-b', '--bsrestart'],
    help='Bandstructure Calculator params.',
    matcher=asr.matchers.EQUAL,
)
@option('--kptpath', type=str, help='Custom kpoint path.')
@option('--npoints',
        type=int,
        help='Number of points along k-point path.')
@option('--kptdensity', help='K-point density', type=float)
@option('--ecut', help='Plane wave cutoff', type=float)
@option('--mode', help='GW mode',
        type=Choice(['G0W0', 'GWG']))
@option('-c', '--correctgw', is_flag=True, default=False)
@option('-z', '--empz', type=float, default=0.75,
        help='Replacement Z for unphysical Zs')
def main(
        atoms: Atoms,
        calculator: dict = gw.defaults.calculator,
        bsrestart: dict = bsmain.defaults.calculator,
        kptpath: dict = bsmain.defaults.kptpath,
        npoints: dict = bsmain.defaults.npoints,
        kptdensity: float = gw.defaults.kptdensity,
        ecut: float = gw.defaults.ecut,
        mode: str = gw.defaults.mode,
        correctgw: bool = True,
        empz: float = 0.75,
) -> Result:
    from gpaw import GPAW
    from asr.utils import fermi_level
    from ase.dft.bandgap import bandgap
    from asr.hse import MP_interpolate
    from types import SimpleNamespace

    gsres = gs(
        atoms=atoms,
        calculator=calculator,
        kptdensity=kptdensity,
        ecut=ecut,
    )
    calc = GPAW(gsres['gs_gw_nowfs.gpw'], txt=None)

    gwresults = empirical_mean_z(
        atoms=atoms,
        calculator=calculator,
        kptdensity=kptdensity,
        ecut=ecut,
        mode=mode,
        correctgw=correctgw,
        empz=empz,
    )
    gwresults = SimpleNamespace(**gwresults)
    lb = gwresults.minband
    ub = gwresults.maxband

    delta_skn = gwresults.qp - gwresults.eps

    # Interpolate band structure
    results = MP_interpolate(
        atoms,
        calculator,
        bsrestart,
        kptpath,
        npoints,
        calc,
        delta_skn,
        lb,
        ub,
    )

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
    theta, phi = get_spin_axis(atoms=atoms, calculator=calculator)
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
