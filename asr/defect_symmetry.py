from ase.io import read
from asr.core import command, option, ASRResult, prepare_result, read_json
from gpaw import restart
import typing
import numpy as np
from pathlib import Path
from ase import Atoms


def get_symmetry_array(sym_results):
    import numpy as np

    Nrows = len(sym_results)
    symmetry_array = np.empty((Nrows, 5), dtype='<U21')
    symmetry_array = np.zeros((Nrows, 5))
    sym_rowlabels = []
    for i, row in enumerate(symmetry_array):
        rowname = sym_results[i]['best']
        sym_rowlabels.append(rowname)
        symmetry_array[i, 0] = sym_results[i]['state']
        symmetry_array[i, 1] = sym_results[i]['spin']
        symmetry_array[i, 2] = sym_results[i]['energy']
        symmetry_array[i, 3] = sym_results[i]['error']
        symmetry_array[i, 4] = sym_results[i]['loc_ratio']

    return symmetry_array, sym_rowlabels


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (WebPanel,
                                      describe_entry,
                                      table,
                                      matrixtable)

    basictable = table(row, 'Defect properties', [
        describe_entry('pointgroup',
                       description=result.key_descriptions['pointgroup'])],
                       key_descriptions, 2)

    symmetry_array, symmetry_rownames = get_symmetry_array(result.symmetries)
    symmetry_table = matrixtable(symmetry_array,
        title='Symmetry label',
        columnlabels=['State', 'Spin', 'Energy [eV]', 'Accuracy', 'Localization ratio'],
        rowlabels=symmetry_rownames)

    summary = {'title': 'Summary',
               'columns': [[basictable], []],
               'sort': 1}

    panel = WebPanel(describe_entry('Defect symmetry (structure and defect states)',
                     description='Structural and electronic symmetry analysis'),
                     columns=[[basictable], [symmetry_table]],
                     sort=3)

    return [panel, summary]


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
class Result(ASRResult):
    """Container for main results for asr.analyze_state."""
    pointgroup: str
    defect_center: typing.Tuple[float, float, float]
    defect_name: str
    symmetries: typing.List[SymmetryResult]

    key_descriptions: typing.Dict[str, str] = dict(
        pointgroup='Point group in Schoenflies notation.',
        defect_center='Position of the defect [Å, Å, Å].',
        defect_name='Name of the defect ({type}_{position})',
        symmetries='List of SymmetryResult objects for all states.'
    )

    formats = {'ase_webpanel': webpanel}


@command(module='asr.defect_symmetry',
         requires=['structure.json', 'unrelaxed.json',
                   '../../unrelaxed.json'],
         resources='1:1h',
         returns=Result)
@option('--mapping/--no-mapping', help='Choose mapping if defect '
        'supercells are created with the general algorithm of '
        'asr.setup.defects, or if non-uniform supercells are used.'
        ' Use --no-mapping otherwise.', is_flag=True)
@option('--radius', help='Radius around the defect where the wavefunction '
        'gets analyzed.', type=float)
def main(mapping: bool = False,
         radius: float = 2.0) -> Result:
    """Analyze defect wavefunctions and their symmetries.

    Note, that you need to set up your folder structure with
    asr.setup.defects in order to correctly run this recipe. Furthermore,
    run asr.get_wfs beforehand to write out the needed wavefunctions."""
    from ase.io.cube import read_cube_data
    from gpaw.point_groups import SymmetryChecker

    # define path of the current directory
    defect = Path('.')

    # check whether input is correct and return important structures
    structure, unrelaxed, primitive, pristine = check_and_return_input()

    # construct mapped structure, or return relaxed defect structure in
    # case mapping is not needed
    if mapping:
        mapped_structure = get_mapped_structure(structure,
                                                unrelaxed,
                                                primitive,
                                                pristine,
                                                defect)
    else:
        mapped_structure = read('structure.json')

    # return point group of the defect structure
    point_group = get_spg_symmetry(mapped_structure)

    # evaluate coordinates of defect in the supercell
    defecttype, defectpos = get_defect_info(primitive, defect)
    defectname = defecttype + '_' + defectpos
    center = return_defect_coordinates(structure,
                                       unrelaxed,
                                       primitive,
                                       pristine,
                                       defect)
    print(f'INFO: defect position: {center}, structural symmetry: {point_group}')

    # symmetry analysis
    checker = SymmetryChecker(point_group, center, radius=radius)
    cubefiles = list(defect.glob('*.cube'))
    if len(cubefiles) == 0:
        raise FileNotFoundError('WARNING: no cube files available in this '
                                'folder!')

    print('spin  band     norm    normcut     best    '
          + ''.join(f'{x:8.3s}' for x in checker.group.symmetries) + 'error')

    labels_up = []
    labels_down = []

    # read in calc once
    atoms, calc = restart('gs.gpw', txt=None)

    symmetry_results = []
    for wf_file in cubefiles:
        spin = str(wf_file)[str(wf_file).find('_') + 1]
        band = str(wf_file)[str(wf_file).find('.') + 1: str(wf_file).find('_')]
        res_wf = find_wf_result(band, spin)
        energy = res_wf['energy']

        wf, atoms = read_cube_data(str(wf_file))
        localization = get_localization_ratio(atoms, wf)

        dct = checker.check_function(wf, (atoms.cell.T / wf.shape).T)
        best = dct['symmetry']
        norm = dct['norm']
        normcut = dct['overlaps'][0]
        error = (np.array(list(dct['characters'].values()))**2).sum()

        print(f'{spin:6} {band:5} {norm:6.3f} {normcut:9.3f} {best:>8}'
              + ''.join(f'{x:8.3f}' for x in dct['characters'].values())
              + '{:9.3}'.format(error))

        [labels_up, labels_down][0].append(best)

        irrep_results = []
        for element in dct['characters']:
            irrep_result = IrrepResult.fromdata(sym_name=element,
                                                sym_score=dct['characters'][element])
            irrep_results.append(irrep_result)

        symmetry_result = SymmetryResult.fromdata(irreps=irrep_results,
                                                  best=best,
                                                  error=error,
                                                  loc_ratio=localization,
                                                  state=band,
                                                  spin=spin,
                                                  energy=energy)
        symmetry_results.append(symmetry_result)

    return Result.fromdata(
        pointgroup=point_group,
        defect_center=center,
        defect_name=defectname,
        symmetries=symmetry_results)


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


def find_wf_result(state, spin):
    """Reads in results of asr.get_wfs and returns WaveFunctionResult
    object that corresponds to the input band and spin index."""
    res = read_json('results-asr.get_wfs.json')
    wfs = res['wfs']
    for wf in wfs:
        if int(wf['state']) == int(state) and int(wf['spin']) == int(spin):
            return wf

    print('ERROR: wf result to given wavefunction file not found! Make sure that '
          'you have consitent wavefunctions or delete old wavefunctions and rerun'
          ' asr.get_wfs.')
    raise Exception('ERROR: can not find corresponging wavefunction result for '
                    f'wavefunction no. {state}/{spin}!')
    return None


def get_mapped_structure(structure, unrelaxed, primitive, pristine, defect):
    """Return centered and mapped structure."""
    threshold = 0.99
    print(primitive)
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
    if not conserved_atoms(ref_struc, primitive, N, defect):
        print('ERROR: number of atoms wrong in {}! Mapping not correct!'.format(
            defect.absolute()))

    return rel_struc


def get_spg_symmetry(structure, symprec=0.1):
    """Returns the symmetry of a given structure evaluated with spglib."""
    import spglib as spg

    spg_sym = spg.get_spacegroup(structure, symprec=symprec, symbol_type=1)

    return spg_sym.split('^')[0]


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
            if (abs(max((artificial.get_positions()[j]
                         - unrelaxed_rattled.get_positions()[i]))) < 0.1
               and i not in indexlist):
                indexlist.append(i)
    for i in range(len(unrelaxed_rattled)):
        if i not in indexlist:
            rmindexlist.append(i)
    return rmindexlist


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
    print(reference)
    N = get_supercell_shape(primitive, pristine)
    reference = reference.repeat((N, N, 1))
    cell = reference.get_cell()
    scell = structure.get_cell()

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
            if not (primitive.get_chemical_symbols()[i]
                    == structure.get_chemical_symbols()[i]):
                label = i
                break
            else:
                label = 0
    elif is_vacancy(defectpath):
        for i in range(len(primitive)):
            if not (primitive.get_chemical_symbols()[i]
                    == structure.get_chemical_symbols()[i]):
                label = i
                break
            else:
                label = 0

    pos = pristine.get_positions()[label]

    return pos


def check_and_return_input():
    """Check whether the folder structure is compatible with this recipe,
    and whether all necessary files are present."""

    pristinepath = list(Path('.').glob('../../defects.pristine*'))[0]
    try:
        pris_struc = read(pristinepath / 'structure.json')
    except FileNotFoundError:
        print('ERROR: pristine structure not available!')
    try:
        struc = read('structure.json')
        unrel = read('unrelaxed.json')
    except FileNotFoundError:
        print('ERROR: defect structure(s) not available!')
    try:
        prim_unrel = read('../../unrelaxed.json')
    except FileNotFoundError:
        print('ERROR: primitive unrelaxed structure not available!')

    print(pris_struc, struc, unrel, prim_unrel)

    return struc, unrel, prim_unrel, pris_struc
