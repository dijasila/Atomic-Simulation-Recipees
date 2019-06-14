import json
from pathlib import Path
from asr.utils import command, option


@command('asr.defect_supercell')
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
    # TBD    
    return structure_sc, N_x, N_y, N_z

def def setup_defects(structure, intrinsic=True, charge_states=3, vacancies=True,
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
    # TBD
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
