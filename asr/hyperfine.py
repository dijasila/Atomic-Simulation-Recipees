import click
import typing
import numpy as np
from pathlib import Path
import ase.units as units

from asr.core import command, read_json, option
from asr.result.resultdata import (HyperfineResult, HFResult, GyromagneticResult,
                                   gyromagnetic_ratios)


# From http://triton.iqfr.csic.es/guide/eNMR/chem/NMRnuclei.html
# Units: MHz/T
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


@command(module='asr.hyperfine',
         requires=['structure.json', 'gs.gpw'],
         dependencies=['asr.gs@calculate'],
         resources='1:1h',
         returns=HFResult)
@option('--center', nargs=3, type=click.Tuple([float, float, float]),
        help='Tuple of three spatial coordinates that should be considered '
        'as the center (defaults to [0, 0, 0]).')
@option('--defect/--no-defect', help='Flag to choose whether HF coupling should be '
        'calculated for a defect. If so, the recipe will automatically extract the '
        'defect position from asr.defect_symmetry.', is_flag=True)
def main(center: typing.Sequence[float] = (0, 0, 0),
         defect: bool = False) -> HFResult:
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

    return HFResult.fromdata(
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
