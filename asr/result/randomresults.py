import typing
import numpy as np
from asr.core import ASRResult, prepare_result
from asr.database.browser import (
    href,
    make_panel_description
)


######### Berry #########
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

######### charge_neutrality #########
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

        if self.off == 0 and self.spin == 0:
            xpos = self.relpos - self.size - shift
            ha = 'right'
        if self.off == 0 and self.spin == 1:
            xpos = self.relpos + self.size + shift
            ha = 'left'
        if self.off == 1 and self.spin == 0:
            xpos = self.relpos - self.size - shift
            ha = 'right'
        if self.off == 1 and self.spin == 1:
            xpos = self.relpos + self.size + shift
            ha = 'left'
        self.ax.text(xpos,
                     self.energy,
                     labelstr,
                     va='center', ha=ha,
                     size=12,
                     color=labelcolor)
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

######### gs #########
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
        from asr.extra_fluff import plot_bs
        data = row.data['results-asr.gw.json']
        return plot_bs(row, filename=filename, bs_label='G₀W₀',
                       data=data,
                       efermi=data['efermi_gw_soc'],
                       cbm=row.get('cbm_gw'),
                       vbm=row.get('vbm_gw'))

######### HSE #########
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
        from asr.extra_fluff import plot_bs
        return plot_bs(row, filename=filename, bs_label='HSE06',
                       data=data,
                       efermi=data['efermi_hse_soc'],
                       vbm=row.get('vbm_hse'),
                       cbm=row.get('cbm_hse'))

######### hyperfine #########
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

######### pdos #########
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

######### sj_analyze #########
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

#########  #########
