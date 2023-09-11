import click
import typing
import numpy as np
from pathlib import Path
import ase.units as units
from ase.geometry import get_distances
from asr.core import (command, ASRResult, prepare_result,
                      read_json, option)
from asr.webpages.browser import make_panel_description
from asr.webpages.appresources import HTMLStringFormat


panel_description = make_panel_description(
    """
    Analysis of hyperfine coupling and spin coherence time.
    """,
    articles=[HTMLStringFormat.href(
        """G. D. Cheng et al. Optical and spin coherence properties of NV
        center in diamond and 3C-SiC, Comp. Mat. Sc. 154, 60 (2018)""",
        'https://doi.org/10.1016/j.commatsci.2018.07.039'),
    ],
)

# From http://triton.iqfr.csic.es/guide/eNMR/chem/NMRnuclei.html
# Units: MHz/T
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
nuclear_abundance = {
    'H': (99.98, 0.5),
    'He': (1.3e-4, 0.5),
    'Li': (92.58, 1.5),
    'Be': (100, 1.5),
    'B': (80.42, 1.5),
    'C': (1.108, 0.5),
    'N': (99.635, 1.0),
    'O': (3.7e-2, 2.5),
    'F': (100, 0.5),
    'Ne': (0.257, 1.5),
    'Na': (100, 0.5),
    'Mg': (10.13, 2.5),
    'Al': (100, 2.5),
    'Si': (4.7, 0.5),
    'P': (100, 0.5),
    'S': (0.76, 1.5),
    'Cl': (75.53, 1.5),
    'K': (93.1, 1.5),
    'Ca': (0.145, 3.5),
    'Sc': (100, 3.5),
    'Ti': (7.28, 2.5),
    'V' : (99.76, 3.5),
    'Cr': (9.55, 1.5),
    'Mn': (100, 2.5),
    'Fe': (2.19, 0.5),
    'Co': (100, 3.5),
    'Ni': (1.19, 1.5),
    'Cu': (69.09, 1.5),
    'Zn': (4.11, 2.5),
    'Ga': (60.4, 1.5),
    'Ge': (7.76, 4.5),
    'As': (100, 1.5),
    'Se': (7.58, 0.5),
    'Br': (50.54, 1.5),
    'Kr': (11.55, 4.5),
    'Rb': (72.15, 2.5),
    'Sr': (7.02, 4.5),
    'Y' : (100, 0.5),
    'Zr': (11.23, 2.5),
    'Nb': (100, 4.5),
    'Mo': (15.72, 2.5),
    'Ru': (17.07, 2.5),
    'Rh': (100, 0.5),
    'Ag': (51.82, 0.5),
    'Cd': (12.75, 0.5),
    'In': (95.72, 4.5),
    'Sn': (8.58, 0.5),
    'Sb': (57.25, 2.5),
    'Te': (6.99, 0.5),
    'I' : (100, 2.5),
    'Xe': (26.44, 0.5),
    'Cs': (100, 3.5),
    'Ba': (11.32, 3.5),
    'Lu': (97.41, 3.5),
    'Hf': (13.75, 4.5),
    'Ta': (99.98, 3.5),
    'W': (14.4, 0.5),
    'Re': (62.93, 2.5),
    'Os': (16.1, 1.5),
    'Ir': (62.7, 1.5),
    'Pt': (33.7, 0.5),
    'Au': (100 , 1.5),
    'Hg': (16.84, 0.5),
    'Tl': (70.5, 0.5),
    'Pb': (22.6 , 0.5),
    'Bi': (100 , 4.5),
    'La': (99.91, 3.5)}
scale = units._e / units._hplanck * 1e-6


class Error(Exception):
    """Base class for other exceptions."""

    pass


class HyperfineNotCalculatedError(Error):
    """Raised when hyperfine tensor could not be calculated."""

    pass


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


def webpanel(result, row, key_description):
    from asr.webpages.browser import (WebPanel,
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
class Result(ASRResult):
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

    formats = {'ase_webpanel': webpanel}


@command(module='asr.hyperfine',
         requires=['structure.json', 'gs.gpw'],
         dependencies=['asr.gs@calculate'],
         resources='1:1h',
         returns=Result)
@option('--center', nargs=3, type=click.Tuple([float, float, float]),
        help='Tuple of three spatial coordinates that should be considered '
        'as the center (defaults to [0, 0, 0]).')
@option('--defect/--no-defect', help='Flag to choose whether HF coupling should be '
        'calculated for a defect. If so, the recipe will automatically extract the '
        'defect position from asr.defect_symmetry.', is_flag=True)
def main(center: typing.Sequence[float] = (0, 0, 0),
         defect: bool = False) -> Result:
    """Calculate hyperfine splitting."""
    from gpaw import GPAW
    from ase.io import read

    # atoms, calc = restart('gs.gpw', txt=None)
    calc = GPAW('gs.gpw', txt=None)
    atoms = read('structure.json')
    hf_results, gfactor_results, ht_int_en, sct = calculate_hyperfine(atoms, calc)

    if defect:
        symmetryresults = 'results-asr.defect_symmetry.json'
        assert Path(symmetryresults).is_file(), (
            'asr.defect_symmetry has to run first!')
        def_res = read_json(symmetryresults)
        center = def_res['defect_center']

    return Result.fromdata(
        hyperfine=hf_results,
        gfactors=gfactor_results,
        center=center,
        delta_E_hyp=ht_int_en,
        sc_time=sct)


def MHz_to_eV(MHz):
    """Convert MHz to eV."""
    J = MHz * 1e6 * units._hplanck

    return J / units._e


def g_factors_from_gyromagnetic_ratios(gyromagnetic_ratios):
    from math import pi

    g_factors = {symbol: ratio * 1e6 * 4 * pi * units._mp / units._e
                 for symbol, (n, ratio) in gyromagnetic_ratios.items()}

    return g_factors


def rescale_hyperfine_tensor(A_avv, g_factors, symbols, magmoms):
    """Rescale hyperfine tensor and diagonalize, return HF results, gyromag. factors."""
    total_magmom = sum(magmoms)
    if not abs(total_magmom) > 0.1:
        raise HyperfineNotCalculatedError('no hyperfine interaction for'
                                          ' zero total mag. moment!')

    g_factor_dict = {}
    hyperfine_results = []
    for a, A_vv in enumerate(A_avv):
        symbol = symbols[a]
        magmom = magmoms[a]
        g_factor = g_factors.get(symbol, 1.0)
        g_factor_dict[symbol] = g_factor
        A_vv *= g_factor / total_magmom * scale
        numbers = np.linalg.eigvalsh(A_vv)
        hyperfine_result = HyperfineResult.fromdata(
            index=a,
            kind=symbol,
            magmom=magmom,
            eigenvalues=numbers)
        hyperfine_results.append(hyperfine_result)

    return hyperfine_results, g_factor_dict


def calculate_hyperfine(atoms, calc):
    """Calculate hyperfine splitting from the calculator."""
    from gpaw.hyperfine import hyperfine_parameters

    # convert from MHz/T to eV
    g_factors = g_factors_from_gyromagnetic_ratios(
        gyromagnetic_ratios)

    # return hyperfine tensor in eV units
    A_avv = hyperfine_parameters(calc)

    magmoms = atoms.get_magnetic_moments()
    symbols = atoms.symbols
    hyperfine_results, g_factor_dict = rescale_hyperfine_tensor(
        A_avv, g_factors, symbols, magmoms)

    gyro_results = GyromagneticResult.fromdict(g_factor_dict)

    # spin coherence time and hyperfine interaction energy to be implemented
    hf_int_en = None
    sct = None

    return hyperfine_results, gyro_results, hf_int_en, sct


if __name__ == '__main__':
    main.cli()
