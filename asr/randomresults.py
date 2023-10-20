import typing
import numpy as np
from asr.core import ASRResult, prepare_result


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
#########  #########

#########  #########

#########  #########

