import typing
import numpy as np
from pathlib import Path
from asr.core import (command, ASRResult, prepare_result,
                      read_json)
from gpaw import restart


def webpanel(result, row, key_description):
    from asr.database.browser import (WebPanel,
                                      describe_entry,
                                      table,
                                      matrixtable)
    hf_results = result.hyperfine
    center = result.center


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


@prepare_result
class Result(ASRResult):
    """Container for asr.hyperfine results."""
    hyperfine: typing.List[HyperfineResult]
    gfactors: typing.List[GyromagneticResult]
    center: typing.Tuple[float, float, float]

    key_descriptions: typing.Dict[str, str] = dict(
        hyperfine='List of HyperfineResult objects for all atoms.',
        gfactors='List of GyromagneticResult objects for each atom species.',
        center='Center to show values on webpanel (only relevant for defects).')


@command(module='asr.hyperfine',
         requires=['structure.json', 'gs.gpw'],
         dependencies=['asr.gs'],
         resources='1:1h',
         returns=Result)
def main() -> Result:
    """Calculate hyperfine splitting."""

    atoms, calc = restart('gs.gpw', txt=None)
    hf_results, gfactor_results = calculate_hyperfine(atoms, calc)

    if Path('results-asr.defect_symmetry.json').is_file():
        def_res = read_json('results-asr.defect_symmetry.json')
        center = def_res['defect_center']
    else:
        center = [None, None, None]

    return Result.fromdata(
        hyperfine=hf_results,
        gfactors=gfactor_results,
        center=center)


def calculate_hyperfine(atoms, calc):
    """Calculate hyperfine splitting from the calculator."""
    from math import pi
    import ase.units as units
    from gpaw.hyperfine import (hyperfine_parameters,
                                gyromagnetic_ratios)

    symbols = atoms.symbols
    magmoms = atoms.get_magnetic_moments()
    total_magmom = atoms.get_magnetic_moment()
    assert total_magmom != 0.0

    g_factors = {symbol: ratio * 1e6 * 4 * pi * units._mp / units._e
                 for symbol, (n, ratio) in gyromagnetic_ratios.items()}

    scale = units._e / units._hplanck * 1e-6
    unit = 'MHz'
    A_avv = hyperfine_parameters(calc)
    print('Hyperfine coupling paramters '
          f'in {unit}:\n')
    columns = ['1.', '2.', '3.']

    print('  atom  magmom      ', '       '.join(columns))

    used = {}
    hyperfine_results = []
    for a, A_vv in enumerate(A_avv):
        symbol = symbols[a]
        magmom = magmoms[a]
        g_factor = g_factors.get(symbol, 1.0)
        used[symbol] = g_factor
        A_vv *= g_factor / total_magmom * scale
        numbers = np.linalg.eigvalsh(A_vv)
        hyperfine_result = HyperfineResult.fromdata(
            index=a,
            kind=symbol,
            magmom=magmom,
            eigenvalues=numbers)
        hyperfine_results.append(hyperfine_result)

        print(f'{a:3} {symbol:>2}  {magmom:6.3f}',
              ''.join(f'{x:9.2f}' for x in numbers))

    print('\nCore correction included')
    print(f'Total magnetic moment: {total_magmom:.3f}')

    print('\nG-factors used:')
    gyro_results = []
    for symbol, g in used.items():
        print(f'{symbol:2} {g:10.3f}')
        gyro_result = GyromagneticResult.fromdata(
            symbol=symbol,
            g=g)
        gyro_results.append(gyro_result)

    return hyperfine_results, gyro_results
