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

    return None


def check_input():
    """Checks if all necessary input files and input parameters for this
    recipe are acessible"""

    # first, get path of 'gs.gpw' file of pristine_sc, as well as the path of
    # 'dielectricconstant.json' of the pristine system
    path_epsilon = find_file_in_folder('dielectricconstant.json', 'pristine')
    path_gs = find_file_in_folder('gs.gpw', 'pristine_sc')

    return None


def find_file_in_folder(filename, foldername):
    """Finds a specific file within a folder starting from your current
    position in the directory tree.
    """
    from pathlib import Path

    p = Path('.')
    tmp_list = list(p.glob('**/' + foldername))

    find_success = False
    if len(tmp_list) == 1:
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
    elif len(tmp_list) == 0:
        print('ERROR: no {0} found in this directory tree'.format(
            foldername))
    else:
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
