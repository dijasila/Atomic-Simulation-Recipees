import typing
from math import pi
import numpy as np
from pathlib import Path
import ase.units as units
from asr.core import (command, ASRResult, prepare_result,
                      read_json)
from asr.database.browser import make_panel_description, href
from gpaw import restart
from gpaw.typing import Array1D
from gpaw.wavefunctions.base import WaveFunctions
from gpaw.hyperfine import expand


panel_description = make_panel_description(
    """
Analysis of hyperfine coupling and spin coherence time.
""",
    articles=[
        href("""G. D. Cheng et al. Optical and spin coherence properties of NV
 center in diamond and 3C-SiC, Comp. Mat. Sc. 154, 60 (2018)""",
             'https://doi.org/10.1016/j.commatsci.2018.07.039'),
    ],
)


def get_atoms_close_to_center(center, row):
    """
    Return ordered list of the atoms closest to the defect.

    Note, that this is the case only if a previous defect calculation is present.
    Return list of atoms closest to the origin otherwise.
    """
    from ase import Atoms
    atoms = row.toatoms()

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
                                      matrixtable,
                                      table,
                                      describe_entry)

    hf_results = result.hyperfine
    center = result.center

    orderarray = get_atoms_close_to_center(center, row)

    hf_array = np.zeros((10, 4))
    hf_atoms = []
    for i, element in enumerate(orderarray[:10, 0]):
        hf_atoms.append(hf_results[int(element)]['kind']
                        + str(hf_results[int(element)]['index']))
        hf_array[i, 0] = f"{hf_results[int(element)]['magmom']:.2f}"
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

    sct_table = table(result, 'Global hyperfine properties', [])
    sct_table['rows'].extend(
        [[describe_entry(f"Hyperfine interaction energy",
                         description=result.key_descriptions['delta_E_hyp']),
          f'{result.delta_E_hyp:.2e} eV'],
         [describe_entry(f"Spin coherence time",
                         description=result.key_descriptions['sc_time']),
          f'{result.sc_time:.2e} ms']])

    hyperfine = WebPanel(describe_entry('HF coupling and spin coherence time',
                                        panel_description),
                         columns=[[hf_table], [gyro_table, sct_table]],
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
def main() -> Result:
    """Calculate hyperfine splitting."""
    atoms, calc = restart('gs.gpw', txt=None)
    hf_results, gfactor_results, ht_int_en, sct = calculate_hyperfine(atoms, calc)

    if Path('results-asr.defect_symmetry.json').is_file():
        def_res = read_json('results-asr.defect_symmetry.json')
        center = def_res['defect_center']
    else:
        center = [None, None, None]

    return Result.fromdata(
        hyperfine=hf_results,
        gfactors=gfactor_results,
        center=center,
        delta_E_hyp=ht_int_en,
        sc_time=sct)


def fermi_contact_interaction_fractions(wfs: WaveFunctions) -> Array1D:
    """Fractions of the defect electron being of the Fermi contact
    interaction with the nuclei.

    See::

        Y. Tu, Z. Tang, X. G. Zhao, Y. Chen, Z. Q. Zhu, J. H. Chu,
        and J. C. Fang: doi.org/10.1063/1.4818659
    """
    # Fine-structure constant: (~1/137)
    alpha = 0.5 * units._mu0 * units._c * units._e**2 / units._hplanck
    assert wfs.world.size == 1

    for kpt_major, kpt_minor in wfs.kpt_qs:
        nocc_major = (kpt_major.f_n > 0.5 * kpt_major.weight).sum()
        nocc_minor = (kpt_minor.f_n > 0.5 * kpt_minor.weight).sum()
        assert nocc_major > nocc_minor

        dN_a = []
        for a, P_ni in kpt_major.projections.items():
            P_i = P_ni[nocc_major - 1]
            D_ii = np.outer(P_i.conj(), P_i)
            setup = wfs.setups[a]
            D_jj = expand(D_ii.real, setup.l_j, l=0)[0]
            phi_jg = np.array(setup.data.phi_jg)
            rgd = setup.rgd
            n_g = np.einsum('ab, ag, bg -> g',
                            D_jj, phi_jg, phi_jg) / (4 * pi)**0.5

            # n(r) = k * r^(-beta)
            beta = 2 * (1 - (1 - (setup.Z * alpha)**2)**0.5)
            k = n_g[1] * rgd.r_g[1]**beta
            r_proton = 0.875e-5 / units.Bohr
            r_nucleus = setup.Z**(1 / 3) * r_proton

            dN_a.append(4 * pi / (3 - beta) * k * r_nucleus**(-beta))

    return np.array(dN_a)


def MHz_to_eV(MHz):
    """Convert MHz to eV."""
    import ase.units as units

    J = MHz * 1e6 * units._hplanck

    return J / units._e


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

    nuclear_abundance = {'H':  (99.98, 0.5),
                         'He':  (1.3*10-4, 0.5),
                         'Li': (92.58, 1.5), 
                         'Be':  (100, 1.5),
                         'B':   (80.42, 1.5),
                         'C': (1.108, 0.5),
                         'N':  (99.635, 1.0), 
                         'O':  (3.7*10-2, 2.5),
                         'F':  (100, 0.5),
                         'Ne':(0.257, 1.5),
                         'Na':  (100, 0.5),
                         'Mg':  (10.13, 2.5),
                         'Al':  (100, 2.5),
                         'Si':  (4.7, 0.5),
                         'P':  (100, 0.5),
                         'S': (0.76, 1.5),
                         'Cl': (75.53, 1.5),
                         'K':  (93.1, 1.5),
                         'Ca': (0.145, 3.5),
                         'Sc': (100,3.5),
                         'Ti':  (7.28, 2.5),
                         'V' :  (99.76, 3.5),
                         'Cr': (9.55, 1.5),
                         'Mn':  (100, 2.5),
                         'Fe':  (2.19, 0.5),
                         'Co':(100, 3.5),
                         'Ni' :  (1.19, 1.5),
                         'Cu':  (69.09, 1.5),
                         'Zn' :  (4.11, 2.5),
                         'Ga':  (60.4, 1.5),
                         'Ge': (7.76, 4.5),
                         'As' :  (100, 1.5),
                         'Se' :  (7.58, 0.5),
                         'Br' :  (50.54, 1.5),
                         'Kr' :  (11.55, 4.5),
                         'Rb' : (72.15, 2.5),
                         'Sr' :  (7.02, 4.5),
                         'Y' : (100, 0.5),
                         'Zr' :  (11.23, 2.5),
                         'Nb' : (100, 4.5),
                         'Mo' : (15.72, 2.5),
                         'Ru' :  (17.07, 2.5),
                         'Rh' :  (100,0.5),
                         'Ag' : (51.82, 0.5),
                         'Cd' : (12.75, 0.5),
                         'In' : (95.72, 4.5),
                         'Sn' :  (8.58, 0.5),
                         'Sb' :  (57.25, 2.5),
                         'Te' :  (6.99, 0.5),
                         'I' :   (100, 2.5),
                         'Xe':  (26.44, 0.5),
                         'Cs' :  (100, 3.5),
                         'Ba' :  (11.32, 3.5),
                         'Lu' :      (97.41, 3.5),
                         'Hf' : (13.75, 4.5), 
                         'Ta' : (99.98, 3.5),
                         'W' : (14.4, 0.5),
                         'Re' :  (62.93, 2.5),
                         'Os':  (16.1, 1.5), 
                         'Ir' : (62.7, 1.5),
                         'Pt' :  (33.7, 0.5),
                         'Au' :  (100 , 1.5),
                         'Hg' : (16.84, 0.5),
                         'Tl' :  (70.5, 0.5),
                         'Pb' :  (22.6 , 0.5),
                         'Bi' :  (100 , 4.5),
                         'La' :  (99.91, 3.5)}

    _hbar = 6.5822e-16 # in eV * s
    _mu_bohr = 5.788381e-5 # in eV / T

    symbols = atoms.symbols
    magmoms = atoms.get_magnetic_moments()
    total_magmom = atoms.get_magnetic_moment()
    assert total_magmom != 0.0

    # convert from MHz/T to eV
    g_factors = {symbol: ratio * 1e6 * 4 * pi * units._mp / units._e
                 for symbol, (n, ratio) in gyromagnetic_ratios.items()}

    scale = units._e / units._hplanck * 1e-6

    # return hyperfine tensor in eV units
    A_avv = hyperfine_parameters(calc)
    print('Hyperfine coupling paramters '
          f'in MHz:\n')
    columns = ['1.', '2.', '3.']
    print('  atom  magmom      ', '       '.join(columns))

    used = {}
    hyperfine_results = []
    hf_list = []
    symbol_list = []
    for a, A_vv in enumerate(A_avv):
        symbol = symbols[a]
        magmom = magmoms[a]
        g_factor = g_factors.get(symbol, 1.0)
        used[symbol] = g_factor
        A_vv *= g_factor / total_magmom * scale
        numbers = np.linalg.eigvalsh(A_vv)
        hf_list.append(sum(numbers) / 3.)
        symbol_list.append(str(symbol))
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

    A = dict(zip(hf_list, symbol_list))
    sym = max(A.items())[1]
    A_max = max(A.items())[0]
    N_nb = 0
    for element in A:
        if element > 0.5 * A_max:
            N_nb += 1

    abundance = nuclear_abundance[sym][0]
    nuclear_spin = nuclear_abundance[sym][1]

    # convert unit of A_max from MHz to eV
    A_max = MHz_to_eV(A_max)

    # hyperfine interaction energy in eV
    wfs = calc.wfs
    hf_int_en = (0.5 * A_max * max(fermi_contact_interaction_fractions(wfs))
                 * (0.01 * abundance) * nuclear_spin)
    print(f'Nuclear spin of {sym} is {nuclear_spin:.2f} with '
          f'nuclear abbundance {abundance:.3f}.')
    print(f'Hyperfine interaction energy: {hf_int_en:.2e} eV')
    sct = 0.5 * _hbar / hf_int_en * 1e3
    print(f'Spin coherence time: {sct:.2e} ms')

    return hyperfine_results, gyro_results, hf_int_en, sct


if __name__ == '__main__':
    main.cli()
