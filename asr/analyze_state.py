from ase.io import read, write
from asr.core import command, option, ASRResult, prepare_result
from gpaw import GPAW, restart
from gpaw.utilities.dipole import dipole_matrix_elements_from_calc
import typing
import numpy as np
from pathlib import Path
from ase import Atoms


def get_atoms_close_to_defect(center):
    """Returns list of the ten atoms closest to the defect."""
    from ase.io import read
    atoms = read('structure.json')

    distancelist = []
    indexlist = []
    ghost_atoms = atoms.copy()
    ghost_atoms.append(Atoms('H', cell=atoms.get_cell(), positions=[center])[0])
    for i, atom in enumerate(ghost_atoms[:-1]):
        meancell = np.mean(atoms.get_cell_lengths_and_angles()[:2])
        distance = ghost_atoms.get_distance(-1, i, mic=True)
        distancelist.append(distance)
        indexlist.append(i)

    orderarray = np.zeros((len(indexlist), 2))
    for i, element in enumerate(indexlist):
        orderarray[i, 0] = element
        orderarray[i, 1] = distancelist[i]
    orderarray = orderarray[orderarray[:, 1].argsort()]

    return orderarray


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (fig, WebPanel, entry_parameter_description,
                                      describe_entry, table, matrixtable)
    import numpy as np
    from ase.io import read

    basictable = table(row, 'Defect info', [
        describe_entry('pointgroup', description=result.key_descriptions['pointgroup'])],
                       key_descriptions, 2)

    hf_results = result.hyperfine
    center = result.defect_center
    orderarray = get_atoms_close_to_defect(center)
    hf_array = np.zeros((10, 4))
    hf_atoms = []
    for i, element in enumerate(orderarray[:10, 0]):
        hf_atoms.append(hf_results[int(element)]['kind'] + str(hf_results[int(element)]['index']))
        hf_array[i, 0] = hf_results[int(element)]['magmom']
        hf_array[i, 1] = hf_results[int(element)]['eigenvalues'][0]
        hf_array[i, 2] = hf_results[int(element)]['eigenvalues'][1]
        hf_array[i, 3] = hf_results[int(element)]['eigenvalues'][2]

    defect_array = np.array([[center[0], center[1], center[2]]])

    hf_table = matrixtable(hf_array,
        title='Atom',
        columnlabels=['Magn. moment', 'Axx (MHz)', 'Ayy (MHz)', 'Azz (MHz)'],
        rowlabels=hf_atoms)

    defect_table = matrixtable(defect_array,
        title=describe_entry('Defect',description='Position of the defect atom.'),
        columnlabels=['x (Å)', 'y (Å)', 'z (Å)'],
        rowlabels=[str(result.defect_name)])

    # rows = basictable['rows']

    # panel = {'title': 'Symmetry',
    #          'columns': [[basictable,
    #                       {'type': 'table', 'header': ['Testtable', ''],
    #                        'rows': [],
    #                        'columnwidth': 4}]],
    #                        'sort': -1}

    summary = {'title': 'Summary',
               'columns': [[basictable], [defect_table]],
               'sort': 1}

    panel = {'title': describe_entry('Symmetry analysis (structure and defect states)', description='Structural and electronic symmetry analysis'),
             'columns': [[basictable]],
             'sort': 2}

    hyperfine = {'title': describe_entry('Hyperfine structure', description='Hyperfine calculations'),
                 'columns': [[hf_table]],
                 'sort': 2}

    return [panel, summary, hyperfine]


@prepare_result
class StatesResult(ASRResult):
    """Container for results of states for one spin channel."""
    states_channel: typing.List[int]
    energies_channel: typing.List[float]
    channel: int

    key_descriptions: typing.Dict[str, str] = dict(
        states_channel='List of indices of defect states.',
        energies_channel='List of energies corresponding to the list of states [eV].',
        channel='Spin channel (0 or 1).'
    )


@prepare_result
class CalculateResult(ASRResult):
    """Container for analyze_state results."""
    states: typing.List[StatesResult]
    dipole: np.ndarray
    localization: np.ndarray
    states_above: bool
    states_below: bool

    key_descriptions: typing.Dict[str, str] = dict(
        states='List of StatesResult objects for one or two spin channels.',
        dipole='Dipole matrix-elements [Å]',
        localization='Localization ratio of the wavefunction.',
        states_above='States present above the Fermi level.',
        states_below='States present below the Fermi level.',
    )


@command(module='asr.analyze_state',
         requires=['gs.gpw', 'structure.json',
                   '../../defects.pristine_sc/gs.gpw'],
         resources='24:2h',
         returns=CalculateResult)
@option('--state', help='Specify the specific state (band number) that you '
        'want to consider. Note, that this argument is not used when the '
        'gap state flag is active.', type=int)
@option('--get-gapstates/--dont-get-gapstates', help='Should all of the gap'
        ' states be saved and analyzed? Note, that if gap states are analysed'
        ' the --state option will be neglected.', is_flag=True)
@option('--analyze/--dont-analyze', help='Not only create cube files of '
        'specific states, but also analyze them.', is_flag=True)
def calculate(state: int = 0,
              get_gapstates: bool = False,
              analyze: bool = False) -> CalculateResult:
    """Write out wavefunction and analyze it.

    This recipe reads in an existing gs.gpw file and writes out wavefunctions
    of different states (either the one of a specific given bandindex or of
    all the defect states in the gap). Furthermore, it will feature some post
    analysis on those states.

    Test.
    """
    atoms = read('structure.json')
    print('INFO: run fixdensity calculation')
    calc = GPAW('gs.gpw', txt='analyze_states.txt')
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})
    if get_gapstates:
        print('INFO: evaluate gapstates ...')
        states, states_above, states_below = return_gapstates_fix(calc, spin=0)
    elif not get_gapstates:
        states = [state]
        states_above = False
        states_below = False

    local_ratio_n = []

    energies_0 = []
    energies_1 = []
    print('INFO: write wavefunctions of gapstates ...')
    for band in states:
        wf = calc.get_pseudo_wave_function(band=band, spin=0)
        energy = calc.get_potential_energy()
        energies_0.append(energy)
        fname = 'wf.{0}_{1}.cube'.format(band, 0)
        write(fname, atoms, data=wf)
        local_ratio_n.append(get_localization_ratio(atoms, wf))
        if calc.get_number_of_spins() == 2:
            wf = calc.get_pseudo_wave_function(band=band, spin=1)
            energy = calc.get_potential_energy()
            energies_1.append(energy)
            fname = 'wf.{0}_{1}.cube'.format(band, 1)
            write(fname, atoms, data=wf)

    states_result_1 = return_states_result(states, energies_0, spin=0)
    states_results = [states_result_1]
    if calc.get_number_of_spins() == 2:
        states_result_2 = return_states_result(states, energies_1, spin=1)
        states_results.append(states_result_2)

    print('INFO: Calculating dipole matrix elements among gap states.')
    d_svnm = dipole_matrix_elements_from_calc(calc, n1=states[0], n2=states[-1] + 1)

    if analyze:
        # To be implemented
        print('INFO: analyze chosen states.')

    return CalculateResult.fromdata(
        states=states_results,
        dipole=d_svnm,
        localization=local_ratio_n,
        states_above=states_above,
        states_below=states_below)


def return_states_result(states, energies, spin) -> StatesResult:
    return StatesResult.fromdata(
        states_channel=states,
        energies_channel=energies,
        channel=spin)


def get_supercell_shape(primitive, pristine):
    """
    Calculates which (NxNx1) supercell would be closest to the given supercell
    created by the general algorithm with respect to number of atoms.

    Returns: N
    """
    N = len(pristine) / len(primitive)
    N = int(np.floor(np.sqrt(N)))
    reconstruct = primitive.copy()
    reconstruct = reconstruct.repeat((N, N, 1))
    rcell = reconstruct.get_cell()
    pcell = pristine.get_cell()
    if rcell[1, 1] > pcell[1, 1]:
        N -= 1
    return N


def get_defect_info(primitive, defectpath):
    """Return defecttype, and kind."""
    defecttype = str(defectpath.absolute()).split(
        '/')[-2].split('_')[-2].split('.')[-1]
    defectpos = str(defectpath.absolute()).split(
        '/')[-2].split('_')[-1]

    return defecttype, defectpos


def return_defect_coordinates(structure, unrelaxed, primitive, pristine,
                              defectpath):
    """Returns the coordinates of the present defect."""
    deftype, defpos = get_defect_info(primitive, defectpath)
    if not is_vacancy(defectpath):
        for i in range(len(primitive)):
            if not (primitive.get_chemical_symbols()[i] ==
                    structure.get_chemical_symbols()[i]):
                label = i
                break
            else:
                label = 0
    elif is_vacancy(defectpath):
        for i in range(len(primitive)):
            if not (primitive.get_chemical_symbols()[i] ==
                    structure.get_chemical_symbols()[i]):
                label = i
                break
            else:
                label = 0

    pos = pristine.get_positions()[label]

    return pos


def recreate_symmetric_cell(structure, unrelaxed, primitive, pristine,
                            translation):
    """
    Function that analyses supercell created by the general algorithm and
    creates symmetric supercell with the atomic positions of the general
    supercell.

    Note: The atoms are not correctly mapped in yet, and also the number
    of atoms is not correct here. It is done in the mapping functions.
    """
    reference = primitive.copy()
    N = get_supercell_shape(primitive, pristine)
    reference = reference.repeat((N, N, 1))
    cell = reference.get_cell()
    scell = structure.get_cell()
    # pcell = primitive.get_cell()

    # create intermediate big structure for the relaxed structure
    bigatoms_rel = structure.repeat((5, 5, 1))
    positions = bigatoms_rel.get_positions()
    positions += [-translation[0], -translation[1], 0]
    positions += -2.0 * scell[0] - 1.0 * scell[1]
    positions += 0.5 * cell[0] + 0.5 * cell[1]
    kinds = bigatoms_rel.get_chemical_symbols()
    rel_struc = Atoms(symbols=kinds, positions=positions, cell=cell)

    # create intermediate big structure for the unrelaxed structure
    bigatoms_rel = unrelaxed.repeat((5, 5, 1))
    positions = bigatoms_rel.get_positions()
    positions += [-translation[0], -translation[1], 0]
    positions += -2.0 * scell[0] - 1.0 * scell[1]
    positions += 0.5 * cell[0] + 0.5 * cell[1]
    kinds = bigatoms_rel.get_chemical_symbols()
    ref_struc = Atoms(symbols=kinds, positions=positions, cell=cell)

    refpos = reference.get_positions()
    refpos += [-translation[0], -translation[1], 0]
    refpos += 0.5 * cell[0] + 0.5 * cell[1]
    reference.set_positions(refpos)
    reference.wrap()

    return rel_struc, ref_struc, reference, cell, N


def is_vacancy(defectpath):
    """
    Checks whether the current defect is a vacancy or substitutional defect.
    Returns true if it is a vacancy, false if it is a substitutional defect.
    """
    try:
        defecttype = str(defectpath.absolute()).split(
            '/')[-2].split('_')[-2].split('.')[-1]
        if defecttype == 'v':
            return True
        else:
            return False
    except IndexError:
        return False


def conserved_atoms(ref_struc, primitive, N, defectpath):
    """
    Returns True if number of atoms is correct after the mapping,
    False if the number is not conserved.
    """
    if (is_vacancy(defectpath) and len(ref_struc) != (N * N * len(primitive) - 1)):
        return False
    elif (not is_vacancy(defectpath) and len(ref_struc) != (N * N * len(primitive))):
        return False
    else:
        print('INFO: number of atoms correct in {}'.format(
            defectpath.absolute()))
        return True


def remove_atoms(structure, indexlist):
    indices = np.array(indexlist)
    indices = np.sort(indices)[::-1]
    for element in indices:
        structure.pop(element)
    return structure


def indexlist_cut_atoms(structure, threshold):
    indexlist = []
    for i in range(len(structure)):
        pos = structure.get_scaled_positions()[i]
        # save indices that are outside the new cell
        if abs(max(pos) > threshold) or min(pos) < -0.01:
            indexlist.append(i)

    return indexlist


def compare_structures(artificial, unrelaxed_rattled):
    indexlist = []
    rmindexlist = []
    for i in range(len(unrelaxed_rattled)):
        for j in range(len(artificial)):
            if (abs(max((artificial.get_positions()[j] -
                         unrelaxed_rattled.get_positions()[i]))) < 0.1
               and i not in indexlist):
                indexlist.append(i)
    for i in range(len(unrelaxed_rattled)):
        if i not in indexlist:
            rmindexlist.append(i)
    return rmindexlist


def get_mapped_structure():
    """Return centered and mapped structure."""
    defect = Path('.')
    unrelaxed = read('unrelaxed.json')
    structure = read('structure.json')
    primitive = read('../../unrelaxed.json')
    pristine = read('../../defects.pristine_sc/structure.json')
    threshold = 0.99
    translation = return_defect_coordinates(structure, unrelaxed, primitive,
                                            pristine, defect)
    rel_struc, ref_struc, artificial, cell, N = recreate_symmetric_cell(structure,
                                                                        unrelaxed,
                                                                        primitive,
                                                                        pristine,
                                                                        translation)
    indexlist = compare_structures(artificial, ref_struc)
    ref_struc = remove_atoms(ref_struc, indexlist)
    rel_struc = remove_atoms(rel_struc, indexlist)
    indexlist = indexlist_cut_atoms(ref_struc, threshold)
    ref_struc = remove_atoms(ref_struc, indexlist)
    rel_struc = remove_atoms(rel_struc, indexlist)
    # if is_vacancy(defect):
    #     N_prim = len(primitive) - 1
    # else:
    #     N_prim = len(primitive)
    if not conserved_atoms(ref_struc, primitive, N, defect):
        threshold = 1.01
        rel_struc, ref_struc, artificial, cell, N = recreate_symmetric_cell(structure,
                                                                            unrelaxed,
                                                                            primitive,
                                                                            pristine,
                                                                            translation)
        indexlist = compare_structures(artificial, ref_struc)
        ref_struc = remove_atoms(ref_struc, indexlist)
        rel_struc = remove_atoms(rel_struc, indexlist)
        indexlist = indexlist_cut_atoms(ref_struc, threshold)
        ref_struc = remove_atoms(ref_struc, indexlist)
        rel_struc = remove_atoms(rel_struc, indexlist)
        # if is_vacancy(defect):
        #     N_prim = len(primitive) - 1
        # else:
        #     N_prim = len(primitive)
    if not conserved_atoms(ref_struc, primitive, N, defect):
        print('ERROR: number of atoms wrong in {}! Mapping not correct!'.format(
            defect.absolute()))

    return rel_struc


def get_spg_symmetry(structure, symprec=0.1):
    """Returns the symmetry of a given structure evaluated with spglib."""
    import spglib as spg

    spg_sym = spg.get_spacegroup(structure, symprec=symprec, symbol_type=1)

    return spg_sym.split('^')[0]


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
        energy='Energy of specific state wrt. vacuum level [eV].'
    )


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
class Result(ASRResult):
    """Container for main results for asr.analyze_state."""
    pointgroup: str
    defect_center: typing.Tuple[float, float, float]
    defect_name: str
    symmetries: typing.List[SymmetryResult]
    hyperfine: typing.List[HyperfineResult]


    key_descriptions: typing.Dict[str, str] = dict(
        pointgroup='Point group in Schoenflies notation.',
        defect_center='Position of the defect [Å, Å, Å].',
        defect_name='Name of the defect ({type}_{position})',
        symmetries='List of SymmetryResult objects for all states.',
        hyperfine='List of HyperfineResult objects for all atoms.'
    )

    formats = {'ase_webpanel': webpanel}


@command(module='asr.analyze_state',
         requires=['structure.json', 'unrelaxed.json',
                   '../../defects.pristine_sc/structure.json',
                   '../../unrelaxed.json'],
         resources='1:1h',
         returns=Result)
@option('--mapping/--no-mapping', is_flag=True)
@option('--radius', type=float)
@option('--hf/--no-hf', is_flag=True)
def main(mapping: bool = True,
         radius: float = 2.0,
         hf: bool = True) -> Result:
    """Analyze wavefunctions and analyze symmetry."""

    from ase.io.cube import read_cube_data
    from asr.core import read_json
    from gpaw.point_groups import SymmetryChecker

    if mapping:
        mapped_structure = get_mapped_structure()
    else:
        mapped_structure = read('structure.json')

    point_group = get_spg_symmetry(mapped_structure)

    # evaluate coordinates of the defect in the supercell
    structure = read('structure.json')
    unrelaxed = read('unrelaxed.json')
    primitive = read('../../unrelaxed.json')
    pristine = read('../../defects.pristine_sc/structure.json')
    defect = Path('.')
    defecttype, defectpos = get_defect_info(primitive, defect)
    defectname = defecttype + '_' + defectpos
    center = return_defect_coordinates(structure,
                                       unrelaxed,
                                       primitive,
                                       pristine,
                                       defect)

    print('INFO: defect position = {}, structural symmetry: {}.'.format(
        center, point_group))

    checker = SymmetryChecker(point_group, center, radius=radius)

    cubefiles = Path('.').glob('*.cube')

    print('spin  band     norm    normcut     best    ' +
          ''.join(f'{x:8.3s}' for x in checker.group.symmetries) + 'error')

    labels_up = []
    labels_down = []

    # read in calc once
    _, calc = restart('gs.gpw', txt=None)

    # get vacuum level to reference energies
    res_def = read_json('results-asr.gs.json')
    evac = res_def.evac

    symmetry_results = []
    for wf_file in cubefiles:
        spin = str(wf_file)[str(wf_file).find('_') + 1]
        band = str(wf_file)[str(wf_file).find('.') + 1: str(wf_file).find('_')]
        energy = calc.get_eigenvalues(spin=int(spin))[int(band)] - evac

        wf, atoms = read_cube_data(str(wf_file))
        localization = get_localization_ratio(atoms, wf)

        dct = checker.check_function(wf, (atoms.cell.T / wf.shape).T)
        best = dct['symmetry']
        norm = dct['norm']
        normcut = dct['overlaps'][0]
        error = (np.array(list(dct['characters'].values()))**2).sum()

        print(f'{spin:6} {band:5} {norm:6.3f} {normcut:9.3f} {best:>8}' +
              ''.join(f'{x:8.3f}' for x in dct['characters'].values()) + '{:9.3}'.format(error))

        [labels_up, labels_down][0].append(best)

        irrep_results = []
        for element in dct['characters']:
            irrep_result = return_irrep_result(element, dct['characters'][element])
            irrep_results.append(irrep_result)

        symmetry_result = return_symmetry_result(irrep_results,
                                                 best,
                                                 error,
                                                 localization,
                                                 band,
                                                 spin,
                                                 energy)
        symmetry_results.append(symmetry_result)

    if hf:
        print('INFO: calculate hyperfine properties.')
        atoms, calc = restart('gs.gpw', txt=None)
        hf_results = calculate_hyperfine(atoms, calc)

    return Result.fromdata(
        pointgroup=point_group,
        defect_center=center,
        defect_name=defectname,
        symmetries=symmetry_results,
        hyperfine=hf_results)


def return_symmetry_result(irreps, best, error, loc_ratio,
        state, spin, energy) -> SymmetryResult:
    """Returns SymmetryResult for a specific state."""
    return SymmetryResult.fromdata(
        irreps=irreps,
        best=best,
        error=error,
        loc_ratio=loc_ratio,
        state=state,
        spin=spin,
        energy=energy)


def return_irrep_result(sym_name, sym_score) -> IrrepResult:
    """Returns IrrepResult object using a check_function dictionary of
    SymmetryChecker class of GPAW."""
    return IrrepResult.fromdata(
        sym_name=sym_name,
        sym_score=sym_score)


def calculate_hyperfine(atoms, calc):
    "Calculate hyperfine splitting."
    from math import pi
    import numpy as np
    import ase.units as units
    from gpaw.hyperfine import hyperfine_parameters,gyromagnetic_ratios

    symbols = atoms.symbols
    magmoms = atoms.get_magnetic_moments()
    total_magmom = atoms.get_magnetic_moment()
    assert total_magmom != 0.0

    g_factors = {symbol: ratio * 1e6 * 4 * pi * units._mp / units._e
             for symbol, (n, ratio) in gyromagnetic_ratios.items()}

    units == 'MHz'
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
    for symbol, g in used.items():
        print(f'{symbol:2} {g:10.3f}')

    return hyperfine_results


def get_localization_ratio(atoms, wf):
    """Returns the localization ratio of the wavefunction,
       defined as the volume of the cell divided the
       integral of the fourth power of the wavefunction."""

    grid_vectors = (atoms.cell.T / wf.shape).T
    dv = abs(np.linalg.det(grid_vectors))
    V = atoms.get_volume()

    IPR = 1 / ((wf**4).sum() * dv)
    local_ratio = V / IPR

    return local_ratio


def plot_gapstates(row, fname):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    evbm, ecbm, gap = get_band_edge()

    # Draw bands edge
    draw_band_edge(evbm, 'vbm', 'C0', offset=gap / 5, ax=ax)
    draw_band_edge(ecbm, 'cbm', 'C1', offset=gap / 5, ax=ax)
    # Loop over eigenvalues to draw the level
    calc = GPAW('gs.gpw')
    nband = calc.get_number_of_bands()
    ef = calc.get_fermi_level()

    for s in range(calc.get_number_of_spins()):
        for n in range(nband):
            ene = calc.get_eigenvalues(spin=s, kpt=0)[n]
            occ = calc.get_occupation_numbers(spin=s, kpt=0)[n]
            enenew = calc.get_eigenvalues(spin=s, kpt=0)[n + 1]
            print(n, ene, occ)
            lev = Level(ene, ax=ax)
            if (ene >= evbm + 0.05 and ene <= ecbm - 0.05):
                # check degeneracy
                if abs(enenew - ene) <= 0.01:
                    lev.draw(spin=s, deg=2)
                elif abs(eneold - ene) <= 0.01:
                    continue
                else:
                    lev.draw(spin=s, deg=1)
                # add arrow if occupied
                if ene <= ef:
                    lev.add_occupation(length=gap / 10)
            if ene >= ecbm:
                break
            eneold = ene

    # plotting
    ax.plot([0, 1], [ef] * 2, '--k')
    ax.set_xlim(0, 1)
    ax.set_ylim(evbm - gap / 5, ecbm + gap / 5)
    ax.set_xticks([])
    ax.set_ylabel('Energy (eV)', size=15)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def get_band_edge():
    calc = GPAW('../../defects.pristine_sc/gs.gpw')
    gap, p1, p2 = bandgap(calc)
    evbm = calc.get_eigenvalues(spin=p1[0], kpt=p1[1])[p1[2]]
    ecbm = calc.get_eigenvalues(spin=p2[0], kpt=p2[1])[p2[2]]
    return evbm, ecbm, gap


def draw_band_edge(energy, edge, color, offset=2, ax=None):
    if edge == 'vbm':
        eoffset = energy - offset
        elabel = energy - offset / 2
    elif edge == 'cbm':
        eoffset = energy + offset
        elabel = energy + offset / 2

    ax.plot([0, 1], [energy] * 2, color=color, lw=2, zorder=1)
    ax.fill_between([0, 1], [energy] * 2, [eoffset] * 2, color=color, alpha=0.7)
    ax.text(0.5, elabel, edge.upper(), color='w', fontsize=18, ha='center', va='center')


class Level:
    """Class to draw a single defect state level in the gap, with an
     arrow if occupied. The direction of the arrow depends on the
     spin channel"""

    def __init__(self, energy, size=0.05, ax=None):
        self.size = size
        self.energy = energy
        self.ax = ax

    def draw(self, spin, deg):
        """Draw the defect state according to its
           spin  and degeneracy"""

        relpos = [[1 / 4,1 / 8],[3 / 4,5 / 8]][spin][deg - 1]
        pos = [relpos - self.size, relpos + self.size]
        self.relpos = relpos
        self.spin = spin
        self.deg = deg

        if deg == 1:
            self.ax.plot(pos, [self.energy] * 2, '-k')

        if deg == 2:
            newpos = [p + 1 / 4 for p in pos]
            self.ax.plot(pos, [self.energy] * 2, '-k')
            self.ax.plot(newpos, [self.energy] * 2, '-k')

    def add_occupation(self, length):
        "Draw an arrow if the defect state is occupied"

        updown = [1, -1][self.spin]
        self.ax.arrow(self.relpos, self.energy - updown*length/2, 0, updown*length, head_width=0.01, head_length=length/5, fc='k', ec='k')
        if self.deg == 2:
            self.ax.arrow(self.relpos + 1/4, self.energy - updown*length/2, 0, updown*length, head_width=0.01, head_length=length/5, fc='k', ec='k')


def return_gapstates(calc_def, spin=0):
    """Evaluates which states are inside the gap and returns the band indices
    of those states for a given spin channel.
    """
    from asr.core import read_json

    _, calc_pris = restart('../../defects.pristine_sc/gs.gpw', txt=None)
    results_pris = read_json('../../defects.pristine_sc/results-asr.gs.json')
    results_def = read_json('results-asr.gs.json')
    vbm = results_pris['vbm'] - results_pris['evac']
    cbm = results_pris['cbm'] - results_pris['evac']

    es_def = calc_def.get_eigenvalues() - results_def['evac']
    es_pris = calc_pris.get_eigenvalues() - results_pris['evac']

    diff = es_pris[0] - es_def[0]
    states_def = es_def + diff

    statelist = []
    [statelist.append(i) for i, state in enumerate(states_def) if (
        state < cbm and state > vbm)]

    return statelist


def return_gapstates_fix(calc_def, spin=0):
    """HOTFIX until spin-orbit works with ASR!"""

    _, calc_pris = restart('../../defects.pristine_sc/gs.gpw', txt=None)
    evac_pris = calc_pris.get_electrostatic_potential()[0, 0, 0]
    evac_def = calc_def.get_electrostatic_potential()[0, 0, 0]
    ef_def = calc_def.get_fermi_level()

    vbm, cbm = calc_pris.get_homo_lumo() - evac_pris

    es_def = calc_def.get_eigenvalues() - evac_def
    ef_def = ef_def - evac_def
    es_pris = calc_pris.get_eigenvalues() - evac_pris

    diff = es_pris[0] - es_def[0]
    states_def = es_def + diff
    ef_def = ef_def + diff

    states_above = False
    states_below = False
    for state in states_def:
        if state < cbm and state > vbm and state > ef_def:
            states_above = True
        elif state < cbm and state > vbm and state < ef_def:
            states_below = True

    statelist = []
    [statelist.append(i) for i, state in enumerate(states_def) if (
        state < cbm and state > vbm)]

    return statelist, states_above, states_below


if __name__ == '__main__':
    main.cli()
