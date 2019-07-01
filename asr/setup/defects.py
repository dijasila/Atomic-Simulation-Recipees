from pathlib import Path
from asr.utils import command, option

##################################################################
# ToDo: incorporate extrinsic defects (optional)
# ToDo: figure out how to pass on all of the different parameters
#       from the 'params.json' files within each folder
# ToDo: implement postprocessing 'collect_data' and 'webpanel'
# ToDo: improve structure and naming of recipe options
##################################################################

@command('asr.setup.defects')
@option('-a', '--atomfile', type=str,
        help='Atomic structure.',
        default='unrelaxed.json')
@option('-q', '--chargestates', type=int,
        help='Charge states included (-q, ..., +q).',
        default=3)
@option('--maxsize', type=float,
        help='Maximum supercell size in Å.',
        default=8.)
@option('--is2d/--is3d', 
        help='Specifies if input structure in atomfile is 2D or 3D.',
        default=True)
@option('--intrinsic', type=bool,
        help='Specify whether you want to incorporate anti-site defects.',
        default=True)
@option('--vacancies', type=bool,
        help='Specify whether you want to incorporate vacancies.',
        default=True)

def main(atomfile, chargestates, maxsize, is2d, intrinsic, vacancies):
    """
    Sets up defect structures for a given host.                               

    Recipe setting up all possible defects within a reasonable supercell as   
    well as the respective pristine system for a given input structure.       
    Defects include: vacancies, anti-site defects. For a given primitive input
    structure this recipe will create a directory tree in the following way:  
    For the example of MoS2:                                                  
      - There has to be a 'unrelaxed.json' file with the primitive structure  
        of the desired system in the folder you run setup.defects. Let this   
        folder be called 'MoS2_setup'. The tree structure will then look like 
        this:

    .                                                                         
    ├── MoS2_defects_setup                                                    
    │   ├── bulk                                                              
    │   │   ├── params.json                                                   
    │   │   └── unrelaxed.json                                                
    │   ├── defects                                                           
    │   │   ├── MoS2_231.HX_at_0                                              
    │   │   │   ├── charge_0                                                  
    │   │   │   │   ├── params.json                                           
    │   │   │   │   └── unrelaxed.json                                        
    │   │   │   ├── charge_1                                                  
    │   │   │   │   ├── params.json                                           
    │   │   │   │   └── unrelaxed.json                                        
    │   │   │   ├── charge_-1                                                 
    │   │   │   │   ├── params.json                                           
    │   │   │   │   └── unrelaxed.json                                        
    │   │   │   ├── charge_2                                                  
    │   │   │   │   ├── params.json                                           
    │   │   │   │   └── unrelaxed.json                                        
    │   │   │   ├── charge_-2                                                 
    │   │   │   │   ├── params.json                                           
    │   │   │   │   └── unrelaxed.json                                        
    .   .   .   .                                                             
    .   .   .                                                                 
    .   .   .                                                                 
    │   │   ├── MoS2_231.HX_at_1                                              
    │   │   │   ├── charge_0                                                  
    │   │   │   │   ├── params.json                                           
    │   │   │   │   └── unrelaxed.json                                        
    │   │   │   ├── charge_1                                                  
    │   │   │   │   ├── params.json                                           
    │   │   │   │   └── unrelaxed.json                                        
    .   .   .   .                                                             
    .   .   .                                                                 
    .   .   .                                                                 
    │   │   ├── MoS2_231.Mo_at_1                                              
    │   │   │   ├── charge_0                                                  
    │   │   │   │   ├── params.json                                           
    │   │   │   │   └── unrelaxed.json                                        
    │   │   │   ├── charge_1                                                  
    │   │   │   │   ├── params.json                                           
    │   │   │   │   └── unrelaxed.json                                        
    .   .   .   .                                                             
    .   .   .                                                                 
    .   .   .                                                                 
    │   │   └── MoS2_231.S_at_0                                               
    │   │       ├── charge_0                                                  
    .   .       .                                                             
    .   .                                                                     
    .   .                                                                     
    │   └── pristine                                                          
    │       ├── params.json                                                   
    │       └── unrelaxed.json                                                
    ├── results_setup.defects.json                                            
    └── unrelaxed.json                                                        

      - Here, the notation for the defects is the following:                  
        'formula_supercellsize.defect_at_substitutionposition' where 'HX'     
        denotes a vacancy                                                     
      - In the resulting folders you can find the unrelaxed structures, as    
        well as a 'params.json' file which contains the charge states of the  
        different defect structures.
    """
    from ase.io import read

    # first, read input atomic structure and store it in ase's atoms object
    structure = read(atomfile)    
    print('INFO: starting recipe for setting up defect systems of '
          '{} host system.'.format(structure.symbols))

    # set up the different defect systems and store their properties
    # in a dictionary
    structure_dict = setup_defects(structure=structure, intrinsic=intrinsic,
                                   charge_states=chargestates,
                                   vacancies=vacancies, 
                                   max_lattice=maxsize, is_2D=is2d)
    
    # based on this dictionary, create a folder structure for all defects 
    # and respective charge states
    create_folder_structure(structure, structure_dict, chargestates)

    return None 


def setup_supercell(structure, max_lattice, is_2D):
    """
    Sets up the supercell of a given structure depending on a 
    maximum supercell lattice vector length for 2D or 3D structures.

    :param structure: input structure (primitive cell)
    :param max_lattice (float): maximum supercell lattice vector length in Å
    :param is_2D (bool): choose 2D or 3D supercell (is_2D=False)

    :return structure_sc: supercell structure 
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
    if not is_2D:
        for z in range(1, 50):
            struc_temp = structure.repeat((1, 1, z))
            diff = struc_temp.get_distance(0, -1)
            if diff > max_lattice:
                z_size = z - 1
                break
    else:
        z_size = 1

    print('INFO: setting up supercell: ({0}, {1}, {2})'.format(
          x_size, y_size, z_size))
    structure_sc = structure.repeat((x_size, y_size, z_size))

    return structure_sc, x_size, y_size, z_size


def setup_defects(structure, intrinsic, charge_states, vacancies,
                  max_lattice, is_2D):
    """
    Sets up all possible defects (i.e. vacancies, intrinsic anti-sites, 
    extrinsic point defects('extrinsic=True')) for a given structure.

    :param structure: input structure (primitive cell)
    :param intrinsic (bool): incorporate intrinsic point defects
    :param vacancies (bool): incorporate vacancies

    :return structure_dict: dictionary of all possible defect configurations
                            of the given structure with different charge 
                            states. The dictionary is built up in the 
                            following way: see folder structure in 'main()'. 
    """
    import spglib

    # set up artificial array in order to check for equivalent positions later
    cell = (structure.cell.array, structure.get_scaled_positions(), 
            structure.numbers)

    # set up a dictionary 
    structure_dict = {}
    formula = structure.symbols

    # set up bulk system
    parameters = {}
    string = 'bulk'
    parameters['txt'] = '{0}.txt'.format(string)
    parameters['charge'] = 0
    structure_dict[string] = {'structure': structure, 'parameters': parameters}

    # first, find the desired supercell
    pristine, N_x, N_y, N_z = setup_supercell(structure, max_lattice, is_2D)
    parameters = {}
    string = 'pristine'
    # try to make naming compatible with defectformation recipe
    parameters['txt'] = '{0}.txt'.format(string)
    parameters['charge'] = 0
    structure_dict[string] = {'structure': pristine, 'parameters': parameters}

    # incorporate the possible vacancies
    dataset = spglib.get_symmetry_dataset(cell)
    eq_pos = dataset.get('equivalent_atoms')

    finished_list = []
    if vacancies:
        temp_dict = {}
        for i in range(len(structure)):
            if not eq_pos[i] in finished_list:
                vacancy = pristine.copy()
                vacancy.pop(i)
                string = '{0}_{1}{2}{3}.HX_at_{4}'.format(
                         formula, N_x, N_y, N_z, i)
                charge_dict = {}
                for q in range((-1) * charge_states, charge_states + 1):
                    parameters = {}
                    parameters['txt'] = '{0}.charged_{1}'.format(string, q)
                    parameters['charge'] = q
                    charge_string = 'charge_{}'.format(q)
                    charge_dict[charge_string] = {'structure': vacancy,
                                                  'parameters': parameters} 
                    temp_dict[string] = charge_dict
            finished_list.append(eq_pos[i])
        #structure_dict['defects'] = temp_dict
    
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
                        defect = pristine.copy()
                        defect[i].symbol = element
                        string = '{0}_{1}{2}{3}.{4}_at_{5}'.format(
                                 formula, N_x, N_y, N_z, element, i)
                        charge_dict = {}
                        for q in range((-1) * charge_states, charge_states +1):
                            parameters = {}
                            parameters['txt'] = '{0}.charged_{1}'.format(string, q)
                            parameters['charge'] = q
                            charge_string = 'charge_{}'.format(q)
                            charge_dict[charge_string] = {'structure': defect,
                                                          'parameters': parameters}
                            temp_dict[string] = charge_dict
                finished_list.append(eq_pos[i])

    # put together structure dict
    structure_dict['defects'] = temp_dict
                
    # TBD!!!
    print('INFO: setting up {0} different defect supercell systems in '
          'charge states -{1}, ..., +{1}, as well as the pristine and bulk '
          'systems.'.format(len(structure_dict['defects']), charge_states))

    return structure_dict


def create_folder_structure(structure, structure_dict, chargestates):
    """
    Creates a folder for every configuration of the defect supercell in 
    the following way:
        - see example directory tree in 'main()'
        - these each contain two files: 'unrelaxed.json' (the defect 
          supercell structure), 'params.json' (the non-general parameters
          of each system)
        - the content of those folders can then be used to do further 
          processing (e.g. relax the defect structure)
    """
    from ase.io import write
    from asr.utils import write_json

    # first, create parent folder for the parent structure
    try:
        parent_folder = str(structure.symbols) + '_defects_setup'
        Path(parent_folder).mkdir()
    except FileExistsError:
        print('WARNING: parent folder ("{0}") for this structure already '
              f'exists in the directory. Skip creating parent folder '
              f'and continue with sub-directories.'.format(parent_folder))
    
    # create a json file for general parameters that are equivalent for all
    # the different defect systems 
    gen_params = {}
    gen_params['chargestates'] = chargestates
    write_json(parent_folder + '/general_parameters.json', gen_params)

    # then, create a seperate folder for each possible defect
    # configuration of this parent folder
    for element in structure_dict:
        folder_name = parent_folder + '/' + element
        try:
            Path(folder_name).mkdir()
        except FileExistsError:
            print('WARNING: folder ("{0}") already exists in this '
                  f'directory. Skip creating it.'.format(folder_name))
        if structure_dict[element].get('structure') is not None:    
            struc = structure_dict[element].get('structure')
            params = structure_dict[element].get('parameters')
            try:
                write(folder_name + '/unrelaxed.json', struc)
                write_json(folder_name + '/params.json', params)
            except FileExistsError:
                print('WARNING: files already exist inside this folder.')
        else:
            sub_dict = structure_dict[element]
            j = 0
            for sub_element in sub_dict:
                defect_name = [key for key in sub_dict.keys()]
                defect_folder_name = folder_name + '/' + defect_name[j]
                j = j + 1
                try:
                    Path(defect_folder_name).mkdir()
                except FileExistsError:
                    print('WARNING: folder ("{0}") already exists in this '
                          f'directory. Skip creating it.'.format(defect_folder_name))
                for i in range((-1)*chargestates, chargestates + 1):
                    charge_name = 'charge_{}'.format(i)
                    charge_folder_name = defect_folder_name + '/' + charge_name
                    try:
                        Path(charge_folder_name).mkdir()
                    except:
                        print('WARNING: folder ("{0}") already exists in this '
                              f'directory. Skip creating it.'.format(charge_folder_name))
                    struc = sub_dict[sub_element].get(charge_name).get('structure')
                    params = sub_dict[sub_element].get(charge_name).get('parameters')
                    write(charge_folder_name + '/unrelaxed.json', struc)
                    write_json(charge_folder_name + '/params.json', params)

    return None


def collect_data():
    return None


def postprocessing():
    """
    Extract data after running setup.defects recipe.
    """
    
    return None 

#def webpanel(row, key_descriptions):
#    from asr.utils.custom import fig, table
#
#    if 'something' not in row.data:
#        return None, []
#
#    table1 = table(row,
#                   'Property',
#                   ['something'],
#                   kd=key_descriptions)
#    panel = ('Title',
#             [[fig('something.png'), table1]])
#    things = [(create_plot, ['something.png'])]
#    return panel, things


#def create_plot(row, fname):
#    import matplotlib.pyplot as plt
#
#    data = row.data.something
#    fig = plt.figure()
#    ax = fig.gca()
#    ax.plot(data.things)
#    plt.savefig(fname)


group = 'setup'
creates = ['unrelaxed.json', 'params.json']  # what files are created
dependencies = []  # no dependencies
resources = '1:10m'  # 1 core for 10 minutes
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart

if __name__ == '__main__':
    main()
