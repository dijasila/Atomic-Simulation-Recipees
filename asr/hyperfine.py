import typing
import numpy as np
from pathlib import Path
from asr.core import (command, ASRResult, prepare_result,
                      read_json)
from gpaw import restart


def get_atoms_close_to_center(center):
    """
    Return ordered list of the atoms closest to the defect.

    Note, that this is the case only if a previous defect calculation is present.
    Return list of atoms closest to the origin otherwise.
    """
    from ase import Atoms
    from ase.io import read
    atoms = read('structure.json')

    if center[0] is None:
        center = [0., 0., 0.]

    distancelist = []
    indexlist = []
    ghost_atoms = atoms.copy()
    ghost_atoms.append(Atoms('H', cell=atoms.get_cell(), positions=[center])[0])
    for i, atom in enumerate(ghost_atoms[:-1]):
        # meancell = np.mean(atoms.get_cell_lengths_and_angles()[:2])
        distance = ghost_atoms.get_distance(-1, i, mic=True)
        distancelist.append(distance)
        indexlist.append(i)

    orderarray = np.zeros((len(indexlist), 2))
    for i, element in enumerate(indexlist):
        orderarray[i, 0] = element
        orderarray[i, 1] = distancelist[i]
    orderarray = orderarray[orderarray[:, 1].argsort()]

    return orderarray


def get_gyro_array(gfactors_results):
    array = np.zeros((len(gfactors_results), 1))
    symbollist = []
    for i, g in enumerate(gfactors_results):
        array[i, 0] = g['g']
        symbollist.append(g['symbol'])

    return array, symbollist


def webpanel(result, row, key_description):
    from asr.database.browser import (WebPanel,
                                      matrixtable)

    hf_results = result.hyperfine
    center = result.center

    orderarray = get_atoms_close_to_center(center)

    hf_array = np.zeros((10, 4))
    hf_atoms = []
    for i, element in enumerate(orderarray[:10, 0]):
        hf_atoms.append(hf_results[int(element)]['kind']
                        + str(hf_results[int(element)]['index']))
        hf_array[i, 0] = f"{int(hf_results[int(element)]['magmom']):2d}"
        hf_array[i, 1] = f"{hf_results[int(element)]['eigenvalues'][0]:.2f}"
        hf_array[i, 2] = f"{hf_results[int(element)]['eigenvalues'][1]:.2f}"
        hf_array[i, 3] = f"{hf_results[int(element)]['eigenvalues'][2]:.2f}"

    hf_table = matrixtable(hf_array,
                           title='Atom',
                           columnlabels=['Magn. moment',
                                         'Axx (MHz)',
                                         'Ayy (MHz)',
                                         'Azz (MHz)'],
                           rowlabels=hf_atoms)

    gyro_array, gyro_rownames = get_gyro_array(result.gfactors)
    gyro_table = matrixtable(gyro_array,
                             title='Symbol',
                             columnlabels=['g-factor'],
                             rowlabels=gyro_rownames)

    hyperfine = WebPanel('Hyperfine structure',
                         columns=[[hf_table], [gyro_table]],
                         sort=1)

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

    formats = {'ase_webpanel': webpanel}


@command(module='asr.hyperfine',
         requires=['structure.json', 'gs.gpw'],
         dependencies=['asr.gs@calculate'],
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
    from gpaw.hyperfine import hyperfine_parameters

    # From http://triton.iqfr.csic.es/guide/eNMR/chem/NMRnuclei.html
    # Units: MHz/T
    gyromagnetic_ratios = {'H': (1, 42.577478),
                           'He': (3, -32.434),
                           'Li': (7, 16.546),
                           'Be': (9, -6.298211),
                           'B':  (11, 13.6611),
                           'C': (13, 10.7084),
                           'N': (14, 3.077),
                           'O': (17, -5.772),
                           'F': (19, 40.052),
                           'Ne':(21, -3.36275),
                           'Na': (23, 11.262),
                           'Mg': (25, -2.6084),
                           'Al': (27, 11.103),
                           'Si': (29, -8.465),
                           'P': (31, 17.235),
                           'S': (33, 3.27045),
                           'Cl':(35, 4.17631),
                           'K': (39, 1.98900),
                           'Ca':(43, -2.86861),
                           'Sc':(45, 10.35739),
                           'Ti': (47, -2.40390),
                           'V' : (51, 11.21232),
                           'Cr': (53, -2.406290),
                           'Mn': (55, 10.5163),
                           'Fe': (57, 1.382),
                           'Co':(59, 10.0532),
                           'Ni' : (61, -3.809960),
                           'Cu': (63, 11.2952439),
                           'Zn' : (67, 2.668563),
                           'Ga': (69, 10.23676),
                           'Ge': (73, -1.48913),
                           'As' : (75, 7.312768),
                           'Se' : (77, 8.14828655),
                           'Br' : (79, 10.69908),
                           'Kr' : (83, -1.64398047),
                           'Rb' : (85, 4.1233194),
                           'Sr' : (89, -1.850870),
                           'Y' : (89, -2.0935685),
                           'Zr' : (91, -3.97213054),
                           'Nb' : (93, 10.44635),
                           'Mo' : (95, 2.7850588),
                           'Ru' : (101, -2.20099224),
                           'Rh' : (103, -1.34637703),
                           'Ag' : (107, -1.7299194),
                           'Cd' : (111, -9.0595),
                           'In' : (115, 9.3749856),
                           'Sn' : (119, -15.9365),
                           'Sb' : (121, 10.2418),
                           'Te' : (125, -13.5242),
                           'I' : (127, 8.56477221),
                           'Xe': (129, -11.8420),
                           'Cs' : (133, 5.614201),
                           'Ba' : (137, 4.755289),
                           'Hf' : (179, -1.08060),
                           'Ta' : (181, 5.1245083),
                           'W' : (183, 1.78243),
                           'Re' : (187,   9.76839),
                           'Os': (189, 1.348764),
                           'Ir' : (193, 0.804325),
                           'Pt' : (195, 9.17955),
                           'Au' : (197, 0.73605), 
                           'Hg' : (199,7.66352), 
                           'Tl' : (205, 24.8093),
                           'Pb' : (207, 8.8167), 
                           'Bi' : (209, 6.91012),
                           'La' : (139, 6.049147)}

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

if __name__ == '__main__':
    main.cli()
