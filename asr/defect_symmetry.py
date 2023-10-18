import numpy as np
import spglib as spg
from pathlib import Path
from ase.io import read
from ase.geometry import wrap_positions

from asr.core import command, option, read_json
from asr.paneldata import DefectSymmetryResult, SymmetryResult, IrrepResult, PristineResult

# TODO: make zrange an input
# TODO: make shift an input


@command(module='asr.defect_symmetry',
         requires=['structure.json'],
         dependencies=['asr.get_wfs'],
         resources='1:6h',
         returns=DefectSymmetryResult)
@option('--primitivefile', help='Path to the primitive structure file.',
        type=str)
@option('--pristinefile', help='Path to the pristine supercell file'
        '(needs to be of the same shape as structure.json).', type=str)
@option('--unrelaxedfile', help='Path to an the unrelaxed '
        'supercell file (only needed if --mapping is set).', type=str)
@option('--mapping/--no-mapping', help='Choose mapping if defect '
        'supercells are created with the general algorithm of '
        'asr.setup.defects, or if non-uniform supercells are used.'
        ' Use --no-mapping otherwise.', is_flag=True)
@option('--radius', help='Radius around the defect where the wavefunction '
        'gets analyzed.', type=float)
def main(primitivefile: str = 'primitive.json',
         pristinefile: str = 'pristine.json',
         unrelaxedfile: str = 'NO',
         mapping: bool = False,
         radius: float = 2.0) -> DefectSymmetryResult:
    """
    Analyze defect wavefunctions and their symmetries.

    Note, that you need to set up your folder structure with
    asr.setup.defects in order to correctly run this recipe. Furthermore,
    run asr.get_wfs beforehand to write out the needed wavefunctions.
    """
    from ase.io.cube import read_cube_data
    from gpaw import restart
    from gpaw.point_groups import SymmetryChecker, point_group_names

    # define path of the current directory, and initialize DefectInfo class
    defectdir = Path('.')
    defectinfo = DefectInfo(defectpath=defectdir)

    # everything where files are handled: input structures, wf_results,
    # calculator and cubefilepaths
    structurefile = 'structure.json'
    structure, unrelaxed, primitive, pristine = check_and_return_input(
        structurefile, unrelaxedfile, primitivefile, pristinefile)
    wf_result = read_json('results-asr.get_wfs.json')
    pris_result = get_pristine_result()
    atoms, calc = restart('gs.gpw', txt=None)
    cubefilepaths = list(defectdir.glob('*.cube'))
    if len(cubefilepaths) == 0:
        raise FileNotFoundError('WARNING: no cube files available in this '
                                'folder!')

    # construct mapped structure, or return relaxed defect structure in
    # case mapping is not needed
    if mapping:
        mapped_structure = get_mapped_structure(structure,
                                                unrelaxed,
                                                primitive,
                                                pristine,
                                                defectinfo)
    else:
        mapped_structure = structure.copy()

    # return point group of the defect structure
    point_group = get_spg_symmetry(mapped_structure)
    print(f'INFO: point group of the defect: {point_group}')

    # loop over cubefiles to save symmetry results
    symmetry_results = []
    centers = []
    for cubefilepath in cubefilepaths:
        cubefilename = str(cubefilepath)
        wfcubefile = WFCubeFile.fromfilename(cubefilename)
        # read cubefile and atoms
        wf, atoms = read_cube_data(wfcubefile.filename)
        # calculate localization ratio
        localization = get_localization_ratio(atoms, wf, calc)
        # evaluate defect center
        Ngrid = calc.get_number_of_grid_points()
        shift = [0.5, 0.5, 0]
        dim = sum(atoms.pbc)
        center = get_defect_center_from_wf(wf=wf, cell=atoms.cell, Ngrid=Ngrid,
                                           shift=shift, dim=dim)
        centers.append(center)
        # extract WF results and energies
        res_wf = find_wf_result(wf_result, wfcubefile.band, wfcubefile.spin)
        energy = res_wf['energy']
        # only evaluate 'best' and 'error' for knows point groups
        if point_group in point_group_names:
            # symmetry analysis only for point groups implemented in GPAW
            checker = SymmetryChecker(point_group, center, radius=radius)
            dct = checker.check_function(wf, (atoms.cell.T / wf.shape).T)
            best = dct['symmetry']
            error = (np.array(list(dct['characters'].values()))**2).sum()
            irrep_results = []
            for element in dct['characters']:
                irrep_result = IrrepResult.fromdata(
                    sym_name=element, sym_score=dct['characters'][element])
                irrep_results.append(irrep_result)
        # otherwise, set irrep results and 'best', 'error' to None
        else:
            irrep_results = [IrrepResult.fromdata(
                sym_name=None,
                sym_score=None)]
            best = None
            error = None

        symmetry_result = SymmetryResult.fromdata(irreps=irrep_results,
                                                  best=best,
                                                  error=error,
                                                  loc_ratio=localization,
                                                  state=wfcubefile.band,
                                                  spin=wfcubefile.spin,
                                                  energy=energy)
        symmetry_results.append(symmetry_result)

    defect_center = average_centers(centers)

    return DefectSymmetryResult.fromdata(
        defect_pointgroup=point_group,
        defect_center=defect_center,
        defect_name=defectinfo.defecttoken,
        symmetries=symmetry_results,
        pristine=pris_result)


def average_centers(centers):
    return np.average(centers, axis=0)


def get_defect_center_from_wf(wf, cell, Ngrid, shift, dim):
    """Extract defect center from individual wavefunction cubefile."""
    if dim == 2:
        midpoint = Ngrid[2] // 2
        zrange = range(midpoint - 5, midpoint + 5)
        print(f'WARNING: {dim}-dimensional structure read in. For the correct '
              'extraction of the defect center, make sure that the structure '
              'is centered along the z-direction of the cell.')
    else:
        zrange = range(Ngrid[2])
    wf_array = get_gridpoints(cell=cell, Ngrid=Ngrid, shift=shift, zrange=zrange)
    density = np.square(wf)
    center = get_center_of_mass(wf_array, density, zrange)
    center -= shift * cell.sum(axis=0)
    # center = shift_positions(center_shifted, shift, cell, invert=True)

    return center


def get_total_mass(m, zrange):
    """Calculate total mass of an array containing weights."""
    mflat = m[:, :, zrange].flatten()
    return np.sum(mflat)


def get_center_of_mass(r, m, zrange):
    """Calculate the center of set of positions r, and weights m."""
    M = get_total_mass(m, zrange)
    coords = [0, 0, 0]
    for i in range(3):
        rflat = r[:, :, zrange, i].flatten()
        mflat = m[:, :, zrange].flatten()
        smd = 0
        for j in range(len(mflat)):
            smd += mflat[j] * rflat[j]
        coords[i] = smd

    return coords / M


def grid_generator(Ngrid, zrange):
    """Yield generator looping over x-, y-, and z-grid."""
    for x in range(Ngrid[0]):
        for y in range(Ngrid[1]):
            for z in zrange:
                yield (x, y, z)


def get_gridpoints(cell, Ngrid, shift, zrange):
    """
    Get an array of grid point coordinates shifted with 'shift'.

    The shape of the array now matches the one containing the wave-
    function weights.
    """
    fullgrid = [Ngrid[0], Ngrid[1], Ngrid[2], 3]
    array = np.zeros(fullgrid)

    lengths = [cell[i] / Ngrid[i] for i in range(3)]
    max_iter_grid = np.prod(Ngrid[:2]) * len(zrange)
    grid_indices = grid_generator(Ngrid, zrange)
    for _ in range(max_iter_grid):
        grid_tuple = next(grid_indices)
        shifts = [grid_tuple[0] * lengths[0][i]
                  + grid_tuple[1] * lengths[1][i]
                  + grid_tuple[2] * lengths[2][i] for i in range(3)]
        shifts += shift * cell.sum(axis=0)
        # shifts = shift_positions(positions, shift, cell)
        wrap = wrap_positions([shifts],
                              cell)
        if wrap[0][0] != shifts[0] or wrap[0][1] != shifts[1]:
            newpos = wrap[0]
        else:
            newpos = shifts
        for i in range(3):
            array[grid_tuple[0], grid_tuple[1], grid_tuple[2], i] = newpos[i]

    return array


def get_spin_and_band(wf_file):
    """Extract spin and band index from cube file name."""
    spin = str(wf_file)[str(wf_file).find('_') + 1]
    band = str(wf_file)[str(wf_file).find('.') + 1: str(wf_file).find('_')]

    return int(spin), int(band)


def get_pristine_result():
    """
    Return PristineResult object.

    In 2D, the reference will be the vacuum level of the pristine calculation.
    In 3D, the reference will be None (vacuum level doesn't make sense here).
    """
    from asr.core import read_json

    try:
        p = Path('.')
        pris = list(p.glob('./../../defects.pristine_sc*'))[0]
        res_pris = read_json(pris / 'results-asr.gs.json')
    except FileNotFoundError as err:
        msg = ('ERROR: does not find pristine results. Did you run setup.defects '
               'and calculate the ground state for the pristine system?')
        raise RuntimeError(msg) from err

    ref_pris = res_pris['evac']
    if ref_pris is None:
        ref_pris = 0

    return PristineResult.fromdata(
        vbm=res_pris['vbm'] - ref_pris,
        cbm=res_pris['cbm'] - ref_pris,
        gap=res_pris['gap'])


def get_localization_ratio(atoms, wf, calc):
    """
    Return the localization ratio of the wavefunction.

    It is defined as the volume of the cell divided the
    integral of the fourth power of the wavefunction.
    """
    assert wf.size == np.prod(calc.wfs.gd.N_c), (
        'grid points in wf cube file and calculator '
        'are not the same!')

    dv = atoms.cell.volume / wf.size
    V = atoms.get_volume()

    IPR = 1 / ((wf**4).sum() * dv)
    local_ratio = V / IPR

    return local_ratio


def find_wf_result(wf_result, state, spin):
    """Read in results of asr.get_wfs and returns WaveFunctionResult."""
    wfs = wf_result['wfs']
    for wf in wfs:
        if wf['state'] == state and wf['spin'] == spin:
            return wf

    raise Exception('ERROR: can not find corresponging wavefunction result for '
                    f'wavefunction no. {state}/{spin}!')


def get_mapped_structure(structure, unrelaxed, primitive, pristine, defectinfo):
    """Return centered and mapped structure."""
    Nvac = defectinfo.number_of_vacancies
    translation = return_defect_coordinates(pristine, defectinfo)
    rel_struc, ref_struc, art_struc, N = recreate_symmetric_cell(
        structure, unrelaxed, primitive, pristine, translation, delta=0)
    for delta in [0.1, 0.3]:
        # for cutoff in [0.01, 0.03, 0.1]:
        for cutoff in np.arange(0.1, 1.2, 0.5):
            rel_tmp = rel_struc.copy()
            ref_tmp = ref_struc.copy()
            art_tmp = art_struc.copy()
            rel_tmp = apply_shift(rel_tmp, delta)
            ref_tmp = apply_shift(ref_tmp, delta)
            art_tmp = apply_shift(art_tmp, delta)
            indexlist = compare_structures(art_tmp, ref_tmp, cutoff)
            del ref_tmp[indexlist]
            del rel_tmp[indexlist]
            for threshold in [1.05, 1.01, 0.99]:
                indexlist = indexlist_cut_atoms(ref_tmp, threshold)
                del ref_tmp[indexlist]
                del rel_tmp[indexlist]
                if conserved_atoms(ref_tmp, primitive, N, Nvac):
                    print(f'Parameters: delta {delta}, '
                          f'cutoff {cutoff}, threshold {threshold}')
                    return rel_tmp

    raise ValueError('number of atoms wrong! Mapping not correct!')


def get_spg_symmetry(structure, symprec=0.1):
    """Return the symmetry of a given structure evaluated with spglib."""
    spgcell = (structure.cell,
               structure.get_scaled_positions(),
               structure.numbers)
    spg_sym = spg.get_spacegroup(spgcell, symprec=symprec, symbol_type=1)

    return spg_sym.split('^')[0]


def conserved_atoms(ref_struc, primitive, N, Nvac):
    """Return whether number of atoms is correct after the mapping or not."""
    if len(ref_struc) == (N * N * len(primitive) - Nvac):
        print('INFO: number of atoms correct after mapping.')
        return True
    else:
        return False


def indexlist_cut_atoms(structure, threshold):
    indexlist = []
    pos = structure.get_scaled_positions(wrap=False)
    for i in range(len(structure)):
        # save indices that are outside the new cell
        if abs(max(pos[i]) > threshold) or min(pos[i]) < 1 - threshold:
            indexlist.append(i)

    return indexlist


def compare_structures(ref_atoms, atoms, cutoff):
    from ase.neighborlist import neighbor_list

    tmp_atoms = atoms + ref_atoms
    nl = neighbor_list('i', tmp_atoms, cutoff=cutoff)
    rmindexlist = []
    for i in range(len(atoms)):
        if i not in nl:
            rmindexlist.append(i)

    return rmindexlist


def recreate_symmetric_cell(structure, unrelaxed, primitive, pristine,
                            translation, delta):
    """
    Recreate a symmetric supercell with atomic positions of the general supercell.

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

    # create intermediate big structure for the relaxed structure
    rel_struc = structure.repeat((5, 5, 1))
    positions = rel_struc.get_positions()
    positions += [-translation[0], -translation[1], 0]
    positions += -2.0 * scell[0] - 1.0 * scell[1]
    # positions += (0.5 + delta) * cell[0] + (0.5 + delta) * cell[1]
    rel_struc.set_positions(positions)
    rel_struc.set_cell(cell)

    # create intermediate big structure for the unrelaxed structure
    ref_struc = unrelaxed.repeat((5, 5, 1))
    positions = ref_struc.get_positions()
    positions += [-translation[0], -translation[1], 0]
    positions += -2.0 * scell[0] - 1.0 * scell[1]
    # positions += (0.5 + delta) * cell[0] + (0.5 + delta) * cell[1]
    ref_struc.set_positions(positions)
    ref_struc.set_cell(cell)

    refpos = reference.get_positions()
    refpos += [-translation[0], -translation[1], 0]
    # refpos += (0.5 + delta) * cell[0] + (0.5 + delta) * cell[1]
    reference.set_positions(refpos)
    reference.wrap()

    return rel_struc, ref_struc, reference, N


def apply_shift(atoms, delta=0):
    newatoms = atoms.copy()
    positions = newatoms.get_positions()
    cell = newatoms.cell
    positions += (0.5 + delta) * cell[0] + (0.5 + delta) * cell[1]
    newatoms.set_positions(positions)
    # scaled_delta = delta / np.mean(atoms.cell.lengths()[:2])
    # newatoms = atoms.copy()
    # spos = newatoms.get_scaled_positions()
    # spos[:2] += 0.5 + scaled_delta
    # newatoms.set_scaled_positions(spos)

    return newatoms


def get_supercell_shape(primitive, pristine):
    """
    Calculate which (NxNx1) supercell would be closest to the given supercell.

    Returns: N
    """
    N = len(pristine) / len(primitive)
    N = int(np.floor(np.sqrt(N)))
    reconstruct = primitive.copy()
    reconstruct = reconstruct.repeat((N, N, 1))
    rcell = reconstruct.get_cell()
    pcell = pristine.get_cell()

    for size in range(N, 0, -1):
        suits = True
        reconstruct = primitive.repeat((size, size, 1))
        rcell = reconstruct.get_cell()
        for i in range(3):
            if rcell[i, i] > pcell[i, i]:
                suits = False
                break
        if suits:
            return size

    return size


class WFCubeFile:
    """Class containing functionalities about WFs and file I/O."""

    def __init__(self, spin, band, wf_data=None, calc=None):
        self.spin = spin
        assert spin in [0, 1], 'spin can only be zero or one!'
        self.band = band
        assert band >= 0, 'negative band indices are not allowed!'
        self.wf_data = wf_data
        self.calc = calc

    @classmethod
    def fromfilename(cls, filename):
        band_spin = filename.split('.')[1]
        band = int(band_spin.split('_')[0])
        spin = int(band_spin.split('_')[1])

        return cls(spin=spin, band=band)

    @property
    def filename(self):
        return f'wf.{self.band}_{self.spin}.cube'

    def write_to_cubefile(self):
        from ase.io import write

        assert (self.wf_data is not None and self.calc is not None), (
            'calculator and wavefunction data needed to write cubefile!')

        write(self.filename, self.calc.atoms, data=self.wf_data)

    def get_wavefunction_from_calc(self):
        assert self.calc is not None, ('initialize WFCubeFile class with a '
                                       'calculator to obtain wavefunction!')
        wf = self.calc.get_pseudo_wave_function(band=self.band, spin=self.spin)
        self.wf_data = wf


class DefectInfo:
    """Class containing all information about a specific defect."""

    def __init__(self,
                 defectpath=None,
                 defecttoken=None):
        assert not (defectpath is None and defecttoken is None), (
            'either defectpath or defecttoken has to be given as input to the '
            'DefectBuilder class!')
        assert not (defectpath is not None and defecttoken is not None), (
            'please give either defectpath or defecttoken as an input, not both!')
        if defectpath is not None:
            self.names, self.specs = self._defects_from_path_or_token(
                defectpath=defectpath)
            self.defecttoken = self._defect_token_from_path(defectpath)
        elif defecttoken is not None:
            self.names, self.specs = self._defects_from_path_or_token(
                defecttoken=defecttoken)

    def _defect_token_from_path(self, defectpath):
        complete_defectpath = Path(defectpath.absolute())
        dirname = complete_defectpath.parent.name
        return ".".join(dirname.split('.')[2:])

    def _defects_from_path_or_token(self, defectpath=None, defecttoken=None):
        """Return defecttype, and kind."""
        if defectpath is not None:
            complete_defectpath = Path(defectpath.absolute())
            dirname = complete_defectpath.parent.name
            defecttoken = dirname.split('.')[2:]
        elif defecttoken is not None:
            defecttoken = defecttoken.split('.')
        if len(defecttoken) >= 2:
            defects = defecttoken[:-1]
            specs_str = defecttoken[-1].split('-')
            specs = [int(spec) for spec in specs_str]
        else:
            defects = defecttoken
            specs = [0]

        return defects, specs

    def get_defect_type_and_kind_from_defectname(self, defectname):
        tokens = defectname.split('_')
        return tokens[0], tokens[1]

    def is_vacancy(self, defectname):
        return defectname.split('_')[0] == 'v'

    def is_interstitial(self, defectname):
        return defectname.split('_')[0] == 'i'

    @property
    def number_of_vacancies(self):
        Nvac = 0
        for name in self.names:
            if self.is_vacancy(name):
                Nvac += 1

        return Nvac


def return_defect_coordinates(pristine, defectinfo):
    """Return the coordinates of the present defect."""
    defect_index = defectinfo.specs[0]
    pos = pristine.get_positions()[defect_index]

    return pos


def check_and_return_input(structurefile='', unrelaxedfile='NO',
                           primitivefile='', pristinefile=''):
    """Check whether all neccessary structures are available."""
    if pristinefile != '':
        try:
            pristine = read(pristinefile)
        except FileNotFoundError as err:
            msg = 'ERROR: pristine structure not available! Check your inputs.'
            raise RuntimeError(msg) from err
    else:
        pristine = None
    if structurefile != '':
        try:
            structure = read(structurefile)
        except FileNotFoundError as err:
            msg = ('ERROR: relaxed defect structure not available! '
                   'Check your inputs.')
            raise RuntimeError(msg) from err
    else:
        structure = None
    if primitivefile != '':
        try:
            primitive = read(primitivefile)
        except FileNotFoundError as err:
            msg = 'ERROR: primitive unrelaxed structure not available!'
            raise RuntimeError(msg) from err
    else:
        primitive = None
    if unrelaxedfile != 'NO':
        try:
            unrelaxed = read(unrelaxedfile)
        except FileNotFoundError as err:
            msg = 'ERROR: unrelaxed defect structure not available! Check your inputs.'
            raise RuntimeError(msg) from err
    else:
        unrelaxed = None

    return structure, unrelaxed, primitive, pristine


if __name__ == '__main__':
    main.cli()
