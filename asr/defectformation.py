from asr.utils import command, option

#############################################################################
#          This recipe is not finished and still under development          #
#############################################################################
# ToDo: include postprocessing functions
# ToDo: get rid of hardcoded sigma, epsilons
# ToDo: read out epsilons from params.json file that is in the bulk folder
#       of the defect setup
# ToDo: add information on system and supercell size in output
# ToDo: get information on Fermi energy for the different formation energies
#############################################################################


@command('asr.defectformation')
@option('--pristine', type=str, default='../../pristine/gs.gpw',
        help='Relative path to ground state .gpw file of pristine host system '
             'on which formation energy calculation is based. Here, the '
             'reference folder is the one with the defects and vacancies '
             'in it, as it was created from setup.defects.')
@option('--defect', type=str, default='gs.gpw',
        help='Ground state .gpw file of disturbed system on which formation '
             'energy calculation is based.')
@option('-q', '--chargestates', type=int,
        help='Charge states included (-q, ..., +q).', default=3)
@option('--is2d/--is3d', default=True, help='Specify wheter you calculate '
                                            'the formation energy in 2D or '
                                            '3D.')
def main(pristine, defect, chargestates, is2d):
    """
    Calculate formation energy of defects.

    This recipe needs the directory structure that was created with
    setup.defects in order to run properly and needs to be launched within
    a particular defect folder, i.e. all of the 'charge_x' folders need to be
    below that folder.
    """
    # from gpaw.defects import ElectrostaticCorrections
    # from asr.utils import read_json
    # import numpy as np
    # from ase.io import read
    q, epsilons, path_gs = check_general_inputs()
    print(q, epsilons, path_gs)

    return None


def check_general_inputs():
    """Checks if all necessary input files and input parameters for this
    recipe are acessible"""
    from asr.utils import read_json

    # first, get path of 'gs.gpw' file of pristine_sc, as well as the path of
    # 'dielectricconstant.json' of the pristine system
    path_epsilon = find_file_in_folder('dielectricconstant.json', 'pristine')
    path_gs = find_file_in_folder('gs.gpw', 'pristine_sc')
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
    if foldername == None:
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
        file_list = list(p.glob(tmp_list[0].name + '/**/' + filename))
        if len(file_list) == 1:
            file_path = file_list[0]
            print('INFO: found {0} of the {1} system: {2}'.format(
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
    return None


def postprocessing():
    return None


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
