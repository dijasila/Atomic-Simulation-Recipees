import json
from pathlib import Path
from asr.utils import command, option

#@command('asr.defect_supercell')
#@option('--number', default=5)
def main(structure):
    """
    Recipe setting up all possible defects within a reasonable
    supercell as well as the respective pristine system for a 
    given input structure. Defects include: vacancies, 
    anti-site defects.
    """
    something = calculate_something(number)
    results = {'number': number,
               'something': something}
    Path('something.json').write_text(json.dumps(results))
    return results


def setup_supercell(structure, max_lattice=6., is_2D=True):
    """
    Sets up the supercell of a given structure depending on a 
    maximum supercell lattice vector length for 2D or 3D structures.

    :param structure: input structure (primitive cell)
    :param max_lattice (float): maximum supercell lattice vector length in Å
    :param is_2D (boolean): choose 2D or 3D supercell

    :return structure_sc: supercell structure 
    """
    for x in range(1, 20):
        struc_temp = structure.repeat((x, 1, 1))
        diff = abs(struc_temp[0].position - struc_temp[-1].position)
        if diff[0] > max_lattice:
            x_size = x - 1
            break
    for y in range(1, 20):
        struc_temp = structure.repeat((1, y, 1))
        diff = abs(struc_temp[0].position - struc_temp[-1].position)
        if diff[1] > max_lattice:
            y_size = y - 1
            break
    if not is_2D:
        for z in range(1, 20):
            struc_temp = structure.repeat((1, 1, z))
            diff = abs(struc_temp[0].position - struc_temp[-1].position)
            if diff[2] > max_lattice:
                z_size = z - 1
                break
    else:
        z_size = 1

    print('Setting up supercell: ({0}, {1}, {2})'.format(x_size, y_size, z_size))
    structure_sc = structure.repeat((x_size, y_size, z_size))

    return structure_sc, x_size, y_size, z_size


def setup_defects(structure, intrinsic=True, charge_states=3, vacancies=True,
                      extrinsic=False, replace_list=None):
    """
    Sets up all possible defects (i.e. vacancies, intrinsic anti-sites, extrinsic
    point defects('extrinsic=True')) for a given structure.

    :param structure: input structure (primitive cell)
    :param intrinsic (boolean): incorporate intrinsic point defects
    :param vacancies (boolean): incorporate vacancies
    :param extrinsic (boolean): incorporate extrinsic point defects
    :param replace_list (array): array of extrinsic dopants and their respective positions

    :return structure_dict: dictionary of all possible defect configurations of the given 
                            structure with different charge states. The dictionary is built 
                            up in the following way: 
                            structure_dict = {'formula_N_x, N_y, N_z.(pristine, vacancy, defect)@
                                               atom_index.charged_(q)': {'structure': defect_structure,
                                               'parameters': parameters},
                                              'formula_ ... : {'structure': ..., 'parameters': ...}}
    """
    import spglib

    # set up artificial array in order to check for equivalent positions later
    cell = (structure.cell.array, structure.get_scaled_positions(), structure.numbers)

    # set up a dictionary 
    structure_dict = {}
    formula = structure.symbols

    # first set up the pristine system by finding the desired supercell
    pristine, N_x, N_y, N_z = setup_supercell(structure)
    parameters = {}
    string = '{0}_{1}{2}{3}.pristine'.format(formula, N_x, N_y, N_z)
    parameters['txt'] = '{0}.txt'.format(string)
    parameters['charge'] = 0
    structure_dict[string] = {'structure': pristine, 'parameters': parameters}

    # incorporate the possible vacancies
    dataset = spglib.get_symmetry_dataset(cell)
    eq_pos = dataset.get('equivalent_atoms')
    finished_list = []
    if vacancies:
        for i in range(len(structure)):
            if not eq_pos[i] in finished_list:
                for q in range ((-1) * charge_states, charge_states + 1):
                    parameters = {}
                    vacancy = pristine.copy()
                    vacancy.pop(i)
                    string = '{0}_{1}{2}{3}.vacancy@{4}.charged_({5})'.format(formula, N_x, N_y, N_z, i, q)
                    parameters['txt'] = '{0}.txt'.format(string)
                    parameters['charge'] = q
                    structure_dict[string] = {'structure': vacancy, 'parameters': parameters}
            finished_list.append(eq_pos[i])

    # incorporate anti-site defects
    finished_list = []
    if intrinsic:
        defect_list = []
        for i in range(len(structure)):
            symbol = structure[i].symbol
            if symbol not in defect_list:
                defect_list.append(symbol)
        for i in range(len(structure)):
            if not eq_pos[i] in finished_list:
                for element in defect_list:
                    if not structure[i].symbol == element:
                        for q in range((-1) * charge_states, charge_states + 1):
                            parameters = {}
                            defect = pristine.copy()
                            defect[i].symbol = element
                            string = '{0}_{1}{2}{3}.defect_{4}@{5}.charged_({6})'.format(formula, N_x, N_y, N_z, element, i, q)
                            parameters['txt'] = '{0}.txt'.format(string)
                            parameters['charge'] = q
                            structure_dict[string] = {'structure': defect, 'parameters': parameters}
                finished_list.append(eq_pos[i])

    # incorporate extrinsic dopants
    # TBD!
#    if extrinsic:
#        for element in replace_list:
#            pass

    return structure_dict


def collect_data(atoms):
    path = Path('something.json')
    if not path.is_file():
        return {}, {}, {}
    # Read data:
    dct = json.loads(path.read_text())
    # Define key-value pairs, key descriptions and data:
    kvp = {'something': dct['something']}
    kd = {'something': ('Something', 'Longer description', 'unit')}
    data = {'something':
            {'stuff': 'more complicated data structures',
             'things': [0, 1, 2, 1, 0]}}
    return kvp, kd, data


def webpanel(row, key_descriptions):
    from asr.utils.custom import fig, table

    if 'something' not in row.data:
        return None, []

    table1 = table(row,
                   'Property',
                   ['something'],
                   kd=key_descriptions)
    panel = ('Title',
             [[fig('something.png'), table1]])
    things = [(create_plot, ['something.png'])]
    return panel, things


def create_plot(row, fname):
    import matplotlib.pyplot as plt

    data = row.data.something
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data.things)
    plt.savefig(fname)


group = 'property'
creates = ['something.json']  # what files are created
dependencies = []  # no dependencies
resources = '1:10m'  # 1 core for 10 minutes
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart

if __name__ == '__main__':
    main()
