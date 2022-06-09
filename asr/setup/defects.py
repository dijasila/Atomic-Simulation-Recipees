"""Generate defective atomic structures."""
import numpy as np
from typing import Sequence
from asr.core import ASRResult

"""
@command('asr.setup.defects')
@option('-a', '--atomfile', type=str,
        help='Atomic structure.')
@option('--supercell', nargs=3, type=click.Tuple([int, int, int]),
        help='List of repetitions in lat. vector directions [N_x, N_y, N_z]')
@option('--maxsize', type=float,
        help='Maximum supercell size in Å.')
@option('--intrinsic', type=bool,
        help='Specify whether you want to incorporate anti-site defects.')
@option('--vacancies', type=bool,
        help='Specify whether you want to incorporate vacancies.')
@option('--double', type=str,
        help='Specify which double defects you want to include. Choose from '
        '"NO", "all", "vac-vac", "vac-sub, sub-sub" as a comma-separated list. E.g. '
        'if you want all vac-vac and vac-sub defects use "vac-vac,vac-sub".')
@option('--double-exclude', type=str,
        help='Comma seperated string with double defects that will be excluded. '
        'E.g. for --double_exclude = "Mg, Fe" all double defects of the types'
        'Mg-Mg, Mg-Fe and Fe-Fe will be excluded from the defect setup. '
        'The string can include both intrinsic and extrinsic elements.')
@option('--scaling-double', type=float,
        help='Scaling factor for double defect creation. All possible double '
        'defects within the sum of covalent radii of the two included sites '
        'times the scaling factor will be generated.')
@option('--uniform-vacuum', type=bool,
        help='If true, tries to set out of plane vacuum size '
        'according to the in plane supercell size. Only for 2D.')
@option('--extrinsic', type=str,
        help='Comma separated string of extrinsic point defect elements.')
@option('--general-algorithm',
        help='Sets up general supercells that break the initial symmetry '
        'of the bravais lattice, as well as choosing the most uniform '
        'configuration with least atoms in the supercell.', type=float)
"""


def main(rn, atoms, supercell: Sequence[int] = (3, 3, 3),
         maxsize: float = None, intrinsic: bool = True, extrinsic: str = 'NO',
         vacancies: bool = True, double: str = 'NO', double_exclude: str = 'NO',
         scaling_double: float = 1.7, uniform_vacuum: bool = False,
         general_algorithm: float = None) -> ASRResult:
    """
    Set up defect structures for a given host.
    Recipe setting up all possible defects within a reasonable supercell as well as the
    respective pristine system for a given input structure. Defects include: vacancies,
    intrinsic substitutional defects. For a given primitive input structure this recipe
    will create a directory tree in the following way (for the example of BN):

    - There has to be a 'unrelaxed.json' file with the primitive structure
      of the desired system in the folder you run setup.defects. The tree
      structure will then look like this:

    MoS2/
    ├── defects.MoS2_331.Mo_S
    │   └── asr.setup.defects.defect-irxvl8uy
    │       ├── atoms.json
    │       ├── input.json
    │       └── output.json
    ├── defects.MoS2_331.S_Mo
    │   └── asr.setup.defects.defect-cd9l_90u
    │       ├── atoms.json
    │       ├── input.json
    │       └── output.json
    ├── defects.MoS2_331.v_Mo
    │   └── asr.setup.defects.defect-jl1ivb5g
    │       ├── atoms.json
    │       ├── input.json
    │       └── output.json
    ├── defects.MoS2_331.v_S
    │   └── asr.setup.defects.defect-3pmt0aic
    │       ├── atoms.json
    │       ├── input.json
    │       └── output.json
    ├── defects.pristine_sc.331
    │   └── asr.setup.defects.defect-xms9yiy1
    │       ├── atoms.json
    │       ├── input.json
    │       └── output.json
    └── define-h7y1gtln
        ├── input.json
        └── output.json

    - Here, the notation for the defects is the following:
      'formula_supercellsize.defect_sustitutionposition' where 'v' denotes a vacancy
    - When the general algorithm is used to set up symmetry broken supercells, the
      foldernames will contain '000' instead of the supersize.
    - In the resulting folders you can find the unrelaxed structures.
    """
    if hasattr(atoms, 'future'):
        if not atoms.future.has_output():
            return
        atoms = atoms.future.value().output
        # Ugly quick fix, at the moment atoms.future.value().output is dict
        if(isinstance(atoms, dict)):
            if('structure' in atoms):
                atoms = atoms['structure']
            else:
                raise ValueError("Input dict atoms need structure key")

    # convert extrinsic defect string
    extrinsic = extrinsic.split(',')

    # convert double defect types string to list
    double = double.split(',')

    # convert double_exclude defect string
    if double_exclude == 'NO':
        double_exclude = frozenset()
    else:
        double_exclude = frozenset(double_exclude.split(','))

    # first, read input atomic structure and store it in ase's atoms object
    structure = atoms

    print('INFO: starting recipe for setting up defect systems of '
          '{} host system.'.format(structure.symbols))
    # check dimensionality of initial parent structure
    nd = sum(structure.pbc)
    if nd == 3:
        is2d = False
    elif nd == 2:
        is2d = True
    elif nd == 1:
        raise NotImplementedError('Setup defects not implemented for 1D '
                                  'structures')
    # set up the different defect systems and store their properties
    # in a dictionary
    structure_dict = setup_defects(structure=structure, intrinsic=intrinsic,
                                   vacancies=vacancies, extrinsic=extrinsic,
                                   double=double, double_exclude=double_exclude,
                                   scaling_factor=scaling_double,
                                   sc=supercell,
                                   max_lattice=maxsize, is_2D=is2d,
                                   vacuum=uniform_vacuum,
                                   general_algorithm=general_algorithm)

    # based on this dictionary, create a folder structure for all defects
    structures = {}

    for element, atoms in structure_dict.items():
        rn2 = rn.with_subdirectory(element)
        structures[element] = rn2.task('asr.setup.defects.defect',
                                       element=element, atoms=atoms)
    return structures


def defect(element, atoms):
    from ase.io import write
    write('unrelaxed.json', atoms)
    return atoms


def setup_supercell(structure, max_lattice, is_2D):
    """Set up the supercell of a given structure.

    Parameters
    ----------
    structure
        input structure (primitive cell)
    max_lattice : float
        maximum supercell lattice vector length in Å
    is_2D : bool
        choose 2D or 3D supercell (is_2D=False)

    Returns
    -------
    structure_sc
        supercell structure
    """
    for x in range(1, 50):
        struc_temp = structure.repeat((x, 1, 1))
        diff = struc_temp.get_distance(0, -1)
        if diff > max_lattice:
            x_size = x - 1
            break
    for y in range(1, 50):
        struc_temp = structure.repeat((1, y, 1))
        diff = struc_temp.get_distance(0, -1)
        if diff > max_lattice:
            y_size = y - 1
            break
    x_size = max(x_size, y_size)
    y_size = x_size
    if not is_2D:
        for z in range(1, 50):
            struc_temp = structure.repeat((1, 1, z))
            diff = struc_temp.get_distance(0, -1)
            if diff > max_lattice:
                z_size = z - 1
                break
        z_size = max(y_size, z_size)
    else:
        z_size = 1

    x_size = max(1, x_size)
    y_size = max(1, y_size)
    z_size = max(1, z_size)
    structure_sc = structure.repeat((x_size, y_size, z_size))

    print('INFO: setting up supercell: ({0}, {1}, {2})'.format(
          x_size, y_size, z_size))

    return structure_sc, x_size, y_size, z_size


def apply_vacuum(atoms):
    """
    Apply vacuum to 2D structures.

    Sets the vacuum automatically for the 2D case (in such a way that
    L_z ~ L_xy).

    :param atoms: input atomic structure

    :return atoms_vac: output atomic structure with changed vacuum size
    """
    atoms_vac = atoms.copy()
    cell = atoms_vac.get_cell()
    oldvac = cell[2][2]
    pos = atoms_vac.get_positions()
    a1 = np.sqrt(cell[0][0]**2 + cell[0][1]**2)
    a2 = np.sqrt(cell[1][0]**2 + cell[1][1]**2)
    a = (a1 + a2) / 2.
    newvac = a
    print('INFO: apply vacuum size to the supercell of the 2D structure '
          'with {} Å.'.format(newvac))
    cell[2][2] = newvac
    pos[:, 2] = pos[:, 2] - oldvac / 2. + newvac / 2.
    atoms_vac.set_cell(cell)
    atoms_vac.set_positions(pos)

    return atoms_vac


def create_vacancies(structure, pristine, eq_pos, base_id):
    """Create vacancy defects, return dictionary of structures and params."""
    defect_dict = {}
    finished_list = []
    for i in range(len(structure)):
        if not eq_pos[i] in finished_list:
            vacancy = pristine.copy()
            sitename = vacancy.symbols[i]
            vacancy.pop(i)
            # rattle defect structure to not get stuck in a saddle point
            vacancy.rattle()
            string = f'defects.{base_id}.v_{sitename}'
            defect_dict[string] = vacancy
        finished_list.append(eq_pos[i])

    return defect_dict


def is_new_double_defect(el1, el2, double_defects):
    """Check whether a new double defect exists already."""
    new = True
    for double in double_defects:
        if el1 in double.split('.') and el2 in double.split('.'):
            new = False

    return new


def is_new_double_defect_2(el1, el2, double_defects, distance, rel_tol=1e-2):
    """Check whether a new double defect exists already."""
    from math import isclose

    for double in double_defects:
        name = double[0]
        ref1 = name.split('.')[0]
        ref2 = name.split('.')[1]
        # elements = name.split('.')
        def_name = f'{ref1}.{ref2}'
        distance_ref = double[1]
        if (el1 != el2 and el1 in name.split('.') and el2 in name.split('.')
           and isclose(distance_ref, distance, rel_tol=rel_tol)):
            return False
        elif (el1 == el2 and f'{el1}.{el2}' == def_name
              and isclose(distance_ref, distance, rel_tol=rel_tol)):
            return False

    return True


def double_defect_index_generator(atoms):
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            if i != j:
                yield (i, j)


def double_defect_species_generator(element_list, defect_type='all',
                                    double_exclude=frozenset()):
    if defect_type == 'all' or defect_type == 'sub-sub':
        for el1 in element_list:
            for el2 in element_list:
                if (not {el1, el2}.issubset(double_exclude)):
                    yield (el1, el2)
    elif defect_type == 'vac-sub':
        for el2 in element_list:
            yield ('v', el2)
    elif defect_type == 'vac-vac':
        yield ('v', 'v')


def get_maximum_distance(atoms, i, j, scaling_factor):
    from ase.data import covalent_radii
    an1 = atoms.numbers[i]
    an2 = atoms.numbers[j]

    R_max = (covalent_radii[an1] + covalent_radii[an2]) * scaling_factor

    return R_max


def create_double_new(structure, pristine, eq_pos,
                      base_id, defect_list=None, scaling_factor=1.5,
                      defect_type='all', double_exclude=frozenset()):
    """Create double defects based on distance criterion."""
    defect_dict = {}
    complex_list = []

    # set up list of all defects considered (intrinsic and extrinsic)
    if defect_list is None:
        defect_list = []
    # get generator object based on the type of defect you are looking for
    defect_list = add_intrinsic_elements(structure, defect_list)
    if defect_type == 'all':
        defect_list.append('v')
        max_iter_elements = len(defect_list) ** 2 - len(double_exclude) ** 2
    elif defect_type == 'sub-sub':
        max_iter_elements = len(defect_list) ** 2 - len(double_exclude) ** 2
    elif defect_type == 'vac-vac':
        defect_list.append('v')
        max_iter_elements = 1
    elif defect_type == 'vac-sub':
        max_iter_elements = len(defect_list)
    double_elements = double_defect_species_generator(defect_list,
                                                      defect_type,
                                                      double_exclude)

    # set up the defects
    max_iter_indices = len(pristine) ** 2 - len(pristine)
    for _ in range(max_iter_elements):
        el1, el2 = next(double_elements)
        double_indices = double_defect_index_generator(pristine)
        for __ in range(max_iter_indices):
            i, j = next(double_indices)
            defect = pristine.copy()
            site1 = f'{el1}_{defect.symbols[i]}'
            site2 = f'{el2}_{defect.symbols[j]}'
            distance = pristine.get_distance(i, j, mic=True)
            R_max = get_maximum_distance(pristine, i, j, scaling_factor)
            if (is_new_double_defect_2(site1, site2,
                                       complex_list, distance)
               and distance < R_max
               and not (el1 == defect.symbols[i] or el2 == defect.symbols[j])):
                defect_string = f'{site1}.{site2}.{i}-{j}'
                complex_list.append((defect_string, distance))
                if el1 == 'v' and el2 == 'v':
                    if i < j:
                        defect.pop(j)
                        defect.pop(i)
                    else:
                        defect.pop(i)
                        defect.pop(j)
                elif el1 == 'v' and el2 != 'v':
                    defect.symbols[j] = el2
                    defect.pop(i)
                elif el2 == 'v' and el1 != 'v':
                    defect.symbols[i] = el1
                    defect.pop(j)
                elif el1 != 'v' and el2 != 'v':
                    defect.symbols[i] = el1
                    defect.symbols[j] = el2
                defect.rattle()
                string = f'defects.{base_id}.{defect_string}'
                defect_dict[string] = defect

    return defect_dict


def create_double(structure, pristine, eq_pos,
                  base_id, defect_list=None):
    """Create double defects."""
    defect_dict = {}
    complex_list = []
    finished_list = []

    # set up list of all defects considered (intrinsic and extrinsic)
    defect_list = add_intrinsic_elements(structure, defect_list)

    print('INFO: create vacancy-vacancy pairs.')
    # vacancy-vacancy pairs
    for i in range(len(structure)):
        if not eq_pos[i] in finished_list:
            for j in range(len(structure)):
                vacancy = pristine.copy()
                site1 = f'v_{vacancy.symbols[i]}'
                site2 = f'v_{vacancy.symbols[j]}'
                if not j == i and is_new_double_defect(site1, site2, complex_list):
                    complex_list.append(f'{site1}.{site2}')
                    if i > j:
                        vacancy.pop(i)
                        vacancy.pop(j)
                    elif j > i:
                        vacancy.pop(j)
                        vacancy.pop(i)
                    # rattle defect structure to not get stuck in a saddle point
                    vacancy.rattle()
                    string = f'defects.{base_id}.{site1}.{site2}'
                    defect_dict[string] = vacancy
                finished_list.append(eq_pos[i])

    print('INFO: create substitutional-substitutional pairs.')
    # substitutional-substitutional pairs
    finished_list = []
    for i in range(len(structure)):
        if not eq_pos[i] in finished_list:
            for element in defect_list:
                for element2 in defect_list:
                    for j in range(len(structure)):
                        defect = pristine.copy()
                        site1 = f'{element}_{defect.symbols[i]}'
                        site2 = f'{element2}_{defect.symbols[j]}'
                        if (not j == i and is_new_double_defect(site1,
                                                                site2,
                                                                complex_list)
                           and not structure[i].symbol == element
                           and not structure[j].symbol == element2):
                            complex_list.append(f'{site1}.{site2}')
                            defect[i].symbol = element
                            defect[j].symbol = element2
                            # rattle defect structure to not get stuck in a saddle point
                            defect.rattle()
                            string = f'defects.{base_id}.{site1}.{site2}'
                            defect_dict[string] = defect
                        finished_list.append(eq_pos[i])

    print('INFO: create vacancy-substitutional pairs.')
    # vacancy-substitutional pairs
    finished_list = []
    for i in range(len(structure)):
        if not eq_pos[i] in finished_list:
            for element in defect_list:
                for j in range(len(structure)):
                    defect = pristine.copy()
                    site1 = f'v_{defect.symbols[i]}'
                    site2 = f'{element}_{defect.symbols[j]}'
                    if (not j == i and is_new_double_defect(site1, site2, complex_list)
                       and not structure[j].symbol == element):
                        complex_list.append(f'{site1}.{site2}')
                        defect[j].symbol = element
                        defect.pop(i)
                        # rattle defect structure to not get stuck in a saddle point
                        defect.rattle()
                        string = f'defects.{base_id}.{site1}.{site2}'
                        defect_dict[string] = defect
                    finished_list.append(eq_pos[i])

    return defect_dict


def add_intrinsic_elements(atoms, elements):
    """Return list of intrinsic elements of a given structure."""
    for i in range(len(atoms)):
        symbol = atoms[i].symbol
        if symbol not in elements:
            elements.append(symbol)

    return elements


def create_substitutional(structure, pristine, eq_pos,
                          base_id, defect_list=None):
    """Create substitutional defects."""
    finished_list = []
    defect_dict = {}

    # get intrinsic doping chemical elements if no input list is given
    if defect_list is None:
        defect_list = add_intrinsic_elements(structure, elements=[])

    for i in range(len(structure)):
        if not eq_pos[i] in finished_list:
            for element in defect_list:
                if not structure[i].symbol == element:
                    defect = pristine.copy()
                    sitename = defect.symbols[i]
                    defect[i].symbol = element
                    # rattle defect structure to not get stuck in a saddle point
                    defect.rattle()
                    string = f'defects.{base_id}.{element}_{sitename}'
                    defect_dict[string] = defect
            finished_list.append(eq_pos[i])

    return defect_dict


def setup_defects(structure, intrinsic, vacancies, extrinsic, double,
                  double_exclude, scaling_factor, sc, max_lattice, is_2D, vacuum,
                  general_algorithm):
    """
    Set up defects for a particular input structure.

    Sets up all possible defects (i.e. vacancies, intrinsic anti-sites,
    extrinsic point defects('extrinsic=True')) for a given structure.

    Parameters
    ----------
    structure
        input structure (primitive cell)
    intrinsic : bool
        incorporate intrinsic point defects
    vacancies : bool
        incorporate vacancies

    Returns
    -------
    structure_dict : dict
        dictionary of all possible defect configurations
        of the given structure. The dictionary is built up in the
        following way: see folder structure in 'main()'.
    """
    import spglib

    # set up artificial array in order to check for equivalent positions later
    cell = (structure.cell.array, structure.get_scaled_positions(),
            structure.numbers)

    # set up a dictionary
    structure_dict = {}
    formula = structure.symbols

    # first, find the desired supercell
    if max_lattice is not None and general_algorithm is False:
        pristine, N_x, N_y, N_z = setup_supercell(
            structure, max_lattice, is_2D)
    elif general_algorithm is not None:
        pristine = create_general_supercell(structure, size=float(general_algorithm))
        N_x = 0
        N_y = 0
        N_z = 0
    else:
        if is_2D:
            N_z = 1
        else:
            N_z = sc[2]
        N_x = sc[0]
        N_y = sc[1]
        print('INFO: setting up supercell: ({0}, {1}, {2})'.format(
              N_x, N_y, N_z))
        pristine = structure.repeat((N_x, N_y, N_z))

    # for 2D structures, adjust vacuum size according to given input
    if is_2D and vacuum:
        pristine = apply_vacuum(pristine)

    parameters = {}
    string = 'defects.pristine_sc.{}{}{}'.format(N_x, N_y, N_z)
    calculator_relax = {}  # relax_calc_dict.copy()
    calculator_gs = {}  # gs_calc_dict.copy()
    parameters['asr.gs@calculate'] = {
        'calculator': calculator_gs}
    parameters['asr.relax'] = {'calculator': calculator_relax}
    structure_dict[string] = pristine

    # incorporate the possible vacancies
    dataset = spglib.get_symmetry_dataset(cell)
    eq_pos = dataset.get('equivalent_atoms')
    base_id = f'{formula}_{N_x}{N_y}{N_z}'

    # defects_dict = {}
    if vacancies:
        defect_dict = create_vacancies(structure,
                                       pristine,
                                       eq_pos,
                                       base_id)
        structure_dict.update(defect_dict)

    # incorporate substitutional defects
    if intrinsic:
        defect_dict = create_substitutional(structure,
                                            pristine,
                                            eq_pos,
                                            base_id)
        structure_dict.update(defect_dict)

    # incorporate extrinsic defects
    if extrinsic != ['NO']:
        defect_list = extrinsic
        defect_dict = create_substitutional(structure,
                                            pristine,
                                            eq_pos,
                                            base_id,
                                            defect_list)
        structure_dict.update(defect_dict)

    # create double defects
    if double != ['NO']:
        if extrinsic != ['NO']:
            defect_list = extrinsic
        else:
            defect_list = None
        for double_type in double:
            defect_dict = create_double_new(structure,
                                            pristine,
                                            eq_pos,
                                            base_id,
                                            defect_list,
                                            scaling_factor,
                                            double_type,
                                            double_exclude)
            structure_dict.update(defect_dict)

    # put together structure dict
    # structure_dict['defects'] = defects_dict

    print('INFO: rattled atoms to make sure defect systems do not get stuck at'
          ' a saddle point.')

    print('INFO: setting up {0} different defect supercell systems '
          ' as well as the pristine supercell '
          'system.'.format(len(structure_dict) - 1))

    return structure_dict


def create_general_supercell(structure, size=12.5):
    """
    Use algorithm to generate general supercell.

    Creates supercell of a form that breaks initial bravais lattice symmetry
    as well as tries to find the most uniform configuration containing the
    least number of atoms. Only works in 2D so far!

    Here's the idea behind the algorithm:
            b1 = n1*a1 + m1*a2
            b2 = n2*a1 + m2*a2
            we restrict ourselves such that m1=0
            the respective new cell is then:
            P = [[n1, 0, 0], [n2, m2, 0], [0, 0, 1]].
    """
    from ase.build import make_supercell
    assert all(structure.pbc == [1, 1, 0]), 'Symmetry breaking only in 2D!'

    # b1 = n1*a1 + m1*a2
    # b2 = n2*a1 + m2*a2
    # we restrict ourselves such that m1=0
    # the respective new cell is then:
    # P = [[n1, 0, 0], [n2, m2, 0], [0, 0, 1]]

    print('INFO: set up general supercell.')
    sc_structuredict = {}
    for n1 in range(1, 10):
        for n2 in range(0, 3):
            for m2 in range(1, 10):
                # set up transformation, only for symmetry broken setup
                if not (n1 == m2 and n2 == 0):
                    P = np.array([[n1, 0, 0], [n2, m2, 0], [0, 0, 1]])
                    sc_structure = make_supercell(prim=structure, P=P)
                    # now implement the postprocessing
                    sc_structuredict[str(n1) + str(0) + str(n2) + str(m2)] = {
                        'structure': sc_structure}
    print('INFO: created all possible linear combinations of the old lattice.')

    # check physical distance between defects
    indexlist = []
    structurelist = []
    numatoms_list = []
    stdev_list = []
    for i, element in enumerate(sc_structuredict):
        cell = sc_structuredict[element]['structure'].get_cell()
        distances = return_distances_cell(cell)
        stdev = np.std(distances)
        sc_structuredict[element]['distances'] = distances
        sc_structuredict[element]['stdev'] = stdev
        minsize = min(distances)
        sc_structuredict[element]['suitable'] = minsize > size
        if minsize > size:
            indexlist.append(i)
            stdev_list.append(stdev)
            numatoms_list.append(len(sc_structuredict[element]['structure']))
            structurelist.append(sc_structuredict[element]['structure'])

    lowlist = []
    lowstruclist = []
    for j, structure in enumerate(structurelist):
        if len(structure) == min(numatoms_list):
            lowlist.append(stdev_list[j])
            lowstruclist.append(structure)
    for k, structure in enumerate(lowstruclist):
        if lowlist[k] == min(lowlist):
            print('INFO: optimal structure found: {}'.format(structure))
            finalstruc = structure

    return finalstruc


def return_distances_cell(cell):
    # there are four possible distinct next neighbor
    # distances of repititions in a given cell

    distances = []
    # calculate a1 and a2 distances
    for i in range(2):
        distances.append(np.sqrt(cell[i][0]**2
                         + cell[i][1]**2
                         + cell[i][2]**2))
    # calculate mixed distances (comb. of a1 and a2)
    for sign in [-1, 1]:
        distances.append(np.sqrt((
            sign * cell[0][0] + cell[1][0])**2 + (
            sign * cell[0][1] + cell[1][1])**2 + (
            sign * cell[0][2] + cell[1][2])**2))

    return distances
