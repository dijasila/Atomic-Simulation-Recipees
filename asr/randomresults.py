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
