from ase.io import read, write
from asr.core import command, option, ASRResult, prepare_result
from gpaw import GPAW, restart
from gpaw.utilities.dipole import dipole_matrix_elements_from_calc
import typing
import numpy as np
from pathlib import Path
from ase import Atoms


# TODO: implement webpanel


@prepare_result
class Result(ASRResult):
    """Container for analyze_state results."""
    states: np.ndarray
    dipole: np.ndarray
    localization: np.ndarray
    states_above: bool
    states_below: bool

    key_descriptions: typing.Dict[str, str] = dict(
        states='List of indices of defect states.',
        dipole='Dipole matrix-elements [Å]',
        localization='Localization ratio of the wavefunction.',
        states_above='States above the Fermi level present.',
        states_below='States below the Fermi level present.'
    )

    # formats = {"ase_webpanel": webpanel}


@command(module='asr.analyze_state',
         requires=['gs.gpw', 'structure.json',
                   '../../defects.pristine_sc/gs.gpw'],
         resources='24:2h',
         returns=Result)
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
              analyze: bool = False) -> Result:
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

    print('INFO: write wavefunctions of gapstates ...')
    for band in states:
        wf = calc.get_pseudo_wave_function(band=band, spin=0)
        fname = 'wf.{0}_{1}.cube'.format(band, 0)
        write(fname, atoms, data=wf)
        local_ratio_n.append(get_localization_ratio(atoms, wf))

        if calc.get_number_of_spins() == 2:
            wf = calc.get_pseudo_wave_function(band=band, spin=1)
            fname = 'wf.{0}_{1}.cube'.format(band, 1)
            write(fname, atoms, data=wf)

    print('INFO: Calculating dipole matrix elements among gap states.')
    d_svnm = dipole_matrix_elements_from_calc(calc, n1=states[0], n2=states[-1] + 1)

    if analyze:
        # To be implemented
        print('INFO: analyze chosen states.')

    return Result.fromdata(
        states=np.array(states),
        dipole=d_svnm,
        localization=local_ratio_n,
        states_above=states_above,
        states_below=states_below)


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


@command(module='asr.analyze_state',
         requires=['structure.json', 'unrelaxed.json',
                   '../../defects.pristine_sc/structure.json',
                   '../../unrelaxed.json'],
         resources='1:1h')
@option('--mapping/--no-mapping', is_flag=True)
@option('--symprec', type=float)
def main(mapping: bool = False,
         symprec: float = 0.1):
    """Analyze wavefunctions and alayze symmetry."""
    # wf_list = return_wavefunction_list()

    if mapping:
        mapped_structure = get_mapped_structure()
    else:
        mapped_structure = read('structure.json')

    spg_sym = get_spg_symmetry(mapped_structure)

    # evaluate coordinates of the defect in the supercell
    structure = read('structure.json')
    unrelaxed = read('unrelaxed.json')
    primitive = read('../../unrelaxed.json')
    pristine = read('../../defects.pristine_sc/structure.json')
    defect = Path('.')
    center = return_defect_coordinates(structure,
                                       unrelaxed,
                                       primitive,
                                       pristine,
                                       defect)

    print('INFO: defect position = {}, structural symmetry: {}.'.format(
        center, spg_sym))


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
