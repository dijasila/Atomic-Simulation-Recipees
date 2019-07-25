from asr.utils import command, option

#############################################################################
#          This recipe is not finished and still under development          #
#############################################################################
# ToDo: include postprocessing functions
# ToDo: add information on system and supercell size in output
# ToDo: testing
#############################################################################


@command('asr.defectformation')
@option('--pristine', type=str, default='gs.gpw',
        help='Name of the groundstate .gpw file of the pristine system. It '
             'always has to be somewhere within a folder that is called '
             '"pristine" in order to work correctly.')
@option('--defect', type=str, default='gs.gpw',
        help='Name of the groundstate .gpw file of the defect systems. They '
             'always have to be within a folder for the specific defect with '
             'a subfolder calles "charge_q" for the respective chargestate q '
             'in order to work correctly.')
@option('--defect_name', default=None,
        help='Runs recipe for all defect folder within your directory when '
             'set to None. Set this option to the name of a desired defect '
             'folder in order for it to run only for this particular defect.')
def main(pristine, defect, defect_name):
    """
    Calculate formation energy of defects.

    This recipe needs the directory structure that was created with
    setup.defects in order to run properly. It has to be launched within the
    folder of the initial structure, i.e. the folder where setup.defects was
    also executed.
    """
    from ase.io import read
    from asr.utils import write_json
    from gpaw import GPAW
    from gpaw.defects import ElectrostaticCorrections
    from pathlib import Path
    import numpy as np
    q, epsilons, path_gs = check_and_get_general_inputs()
    atoms = read('unrelaxed.json')
    nd = int(np.sum(atoms.get_pbc()))

    sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    if nd == 3:
        epsilon = (epsilons[0] + epsilons[1] + epsilons[2])/3.
        dim = '3d'
    elif nd == 2:
        epsilon = [(epsilons[0] + epsilons[1])/2., epsilons[2]]
        dim = '2d'

    folder_list = []
    p = Path('.')

    # Either run for all defect folders or just a specific one
    if defect_name is None:
        [folder_list.append(x) for x in p.iterdir() if x.is_dir()
            and not x.name == 'pristine' and not x.name == 'pristine_sc']
    else:
        [folder_list.append(x) for x in p.iterdir() if x.is_dir()
            and x.name == defect_name]

    defectformation_dict = {}
    for folder in folder_list:
        e_form_name = 'e_form_' + folder.name
        e_form = []
        e_fermi = []
        charges = []
        for charge in range(-q, q + 1):
            tmp_folder_name = folder.name + '/charge_' + str(charge)
            charged_file = find_file_in_folder('unrelaxed.json',
                                               tmp_folder_name)
            # elc = ElectrostaticCorrections(pristine=path_gs,
            #                                charged=charged_file,
            #                                q=charge, sigma=sigma,
            #                                dimensionality=dim)
            # elc.set_epsilons(epsilon)
            # e_form.append(elc.calculate_corrected_formation_energy())
            # calc = GPAW(find_file_in_folder('gs.gpw', tmp_folder_name))
            # e_fermi.append(calc.get_fermi_level())
            charges.append(charge)
        defectformation_dict[folder.name] = {'formation_energies': e_form,
                                             'fermi_energies': e_fermi,
                                             'chargestates': charges}
    write_json('defectformation.json', defectformation_dict)

    return None


def check_and_get_general_inputs():
    """Checks if all necessary input files and input parameters for this
    recipe are acessible"""
    from asr.utils import read_json

    # first, get path of 'gs.gpw' file of pristine_sc, as well as the path of
    # 'dielectricconstant.json' of the pristine system
    path_epsilon = find_file_in_folder('dielectricconstant.json', 'pristine')
    path_gs = find_file_in_folder('gs.gpw', 'pristine_sc/neutral')
    path_q = find_file_in_folder('general_parameters.json', None)

    # if paths were found correctly, extract epsilon and q
    gen_params = read_json(path_q)
    params_eps = read_json(path_epsilon)
    q = gen_params.get('chargestates')
    epsilons = params_eps.get('local_field')
    if q is not None and epsilons is not None:
        msg = 'INFO: number of chargestates and dielectric constant '
        msg += 'extracted: q = {}, eps = {}'.format(q, epsilons)
        print(msg)
    else:
        msg = 'either number of chargestates and/or dielectric '
        msg += 'constant of the host material could not be extracted'
        raise ValueError(msg)

    if path_gs is not None:
        print('INFO: check of general inputs successful')

    return q, epsilons, path_gs


def find_file_in_folder(filename, foldername):
    """Finds a specific file within a folder starting from your current
    position in the directory tree.
    """
    from pathlib import Path

    p = Path('.')
    find_success = False

    # check in current folder directly if no folder specified
    if foldername is None:
        check_empty = True
        tmp_list = list(p.glob(filename))
        if len(tmp_list) == 1:
            file_path = tmp_list[0]
            print('INFO: found {0}: {1}'.format(filename,
                                                file_path.absolute()))
            find_success = True
        else:
            print('ERROR: no unique {} found in this directory'.format(
                filename))
    else:
        tmp_list = list(p.glob('**/' + foldername))
        check_empty = False
    # check sub_folders
    if len(tmp_list) == 1 and not check_empty:
        file_list = list(p.glob(foldername + '/**/' + filename))
        if len(file_list) == 1:
            file_path = file_list[0]
            print('INFO: found {0} in {1}: {2}'.format(
                filename, foldername, file_path.absolute()))
            find_success = True
        elif len(file_list) == 0:
            print('ERROR: no {} found in this directory'.format(
                filename))
        else:
            print('ERROR: several {0} files in directory tree: {1}'.format(
                filename, tmp_list[0].absolute()))
    elif len(tmp_list) == 0 and not check_empty:
        print('ERROR: no {0} found in this directory tree'.format(
            foldername))
    elif not check_empty:
        print('ERROR: several {0} folders in directory tree: {1}'.format(
            foldername, p.absolute()))

    if not find_success:
        file_path = None

    return file_path


def collect_data():
    from ase.io import jsonio
    from pathlib import Path
    if not Path('defectformation.json').is_file():
        return {}, {}, {}

    kvp = {}
    data = {}
    key_descriptions = {}
    dct = jsonio.decode(Path('defectformation.json').read_text())

    # Update key-value-pairs

    return None


def postprocessing():
    from asr.utils import read_json

    formation_dict = read_json('defectformation.json')

    return formation_dict


# def webpanel(row, key_descriptions):
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


# def create_plot(row, fname):
#    import matplotlib.pyplot as plt
#
#    data = row.data.something
#    fig = plt.figure()
#    ax = fig.gca()
#    ax.plot(data.things)
#    plt.savefig(fname)


group = 'property'
creates = []  # what files are created
# dependencies = ['asr.setup.defects', 'asr.relax', 'asr.gs',
#                'asr.polarizability']
resources = '1:10m'  # 1 core for 10 minutes
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart

if __name__ == '__main__':
    main()
