import json
from pathlib import Path
from ase.io import read
from asr.core import command, option
from math import isclose
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

@command('asr.setup.stacking',
         resources='1:1h')
@option('--distance', type=float,
        help='Initial distance between the monolayers')
def main(distance=12.):
    """
    Creates bilayer structures.
    """
    atom = read('structure.json')

    try:
        magstate = get_magstate(atom)
    except RuntimeError:
        magstate = 'nm'

    structure_list, name_list =  setup_rotation(atom, distance)
    create_folder_structure(structure_list, name_list)
    print('INFO: finished!')

def get_magstate(atom):
    """
    Obtains the magnetic state of a given atomic structure
    """
    magmom = atom.get_magnetic_moment()
    magmoms = atom.get_magnetic_moments()

    if abs(magmom) > 0.02:
        state = 'fm'
    elif abs(magmom) < 0.02 and abs(magmoms).max() > 0.1:
        state = 'afm'
    else:
        state = 'nm'

    return state


def setup_rotation(atom, distance):
    """
    Analyzes both cell and basis. Checks first which cell we have and rotates
    the structures accordingly. Afterwards, check whether those rotations are
    equivalent with symmetry operations for the spacegroup. If that's not the
    case, keep the rotations and continue.
    """
    from ase.io import write
    import spglib

    cell = atom.get_cell()
    print('INFO: atoms object: {}'.format(atom))
    bravais = cell.get_bravais_lattice()
    print('INFO: Bravais lattice for this structure: {}'.format(bravais))

    cell_spg = (atom.cell.array, atom.get_scaled_positions(),
                atom.numbers)
    print('Spglib cell: {}'.format(cell_spg))
    dataset = spglib.get_symmetry_dataset(cell_spg)
    print('Spglib dataset: {}'.format(dataset))

    name = bravais.lattice_system
    if name == 'cubic':
        rotations = 90
    else:
        rotations = 180

    i = 0
    rot_list = []
    structure_list = []
    name_list = []
    for rot in range(0, 360, rotations):
        print('INFO: rotation {}: {}'.format(i, rot))
        rot_list.append(rot)
        name = 'stacking.rot_{}.trans_0'.format(rot)
        name_list.append(name)
        i = i + 1
    print('Rotation list: {}'.format(rot_list))
    for el in rot_list:
        print('INFO: applied rotation {}'.format(el))
        newstruc = atom.copy()
        newstruc.rotate(el, 'z', rotate_cell=False)
        newstruc.wrap()

        newpos = newstruc.get_positions()
        newpos[:, 2] = newpos[:, 2] + distance
        newstruc.set_positions(newpos)

        newstruc = newstruc + atom
        structure_list.append(newstruc)

    return structure_list, name_list


def create_folder_structure(structure_list, name_list):
    """
    TBD
    """
    from ase.io import write
    from pathlib import Path
    print('Structures to be created: {}'.format(structure_list))
    print('Folder names to be created: {}'.format(name_list))

    for i in range(len(name_list)):
        print(i)
        Path(name_list[i]).mkdir()
        write(name_list[i]+'/structure.json', structure_list[i])

    print('INFO: finished IO part of the setup recipe')


# ToDo:
# - Compare rotations of the cell with the ones of the spacegroup and only
# keep the ones that are only present for the cell
# - Keep translations in the ToDo list, first figuring out the rotations
# properly is more important
# - Find suitable naming convention for the different folders
# - Put io in separate function
#   * working for rotations now
# - expand name list for figuring out rotations of the cell
# - extract rotation around z from rotation matrix
#   * that one will be extracted from spglibs symmetry dataset


if __name__ == '__main__':
    main()
