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
    from asr.utils import read_json
    # from ase.io import read

    # ToDo: calculate sigma correctly for different systems
    # ToDo: get rid of hardcoded epsilon

    # TBD!!!
    # sigma = 1.0
    # epsilons = [1.9, 1.15]

    # read out general parameters from generals_params.json
    gen_params = read_json('../../general_parameters.json')
    chargestates_read = gen_params.get('chargestates')
    print('INFO: read out general parameters: {}'.format(chargestates_read))

    # get dimensionality of the system
    # if is2d:
    #    dim = '2d'
    #    # epsilons = [x, y]
    # elif is2d == False:
    #    dim = '3d'
    #    # epsilons = x

    # get groundstate file name of the pristine system
    pristine_file = pristine
    print('INFO: use pristine gs file "{}"'.format(pristine_file))

    # first, loop over all charge states and access the right charge state
    # folder with the correct 'gs.gpw'
    # eform_array = []
    q_array = []
    for i in range((-1) * chargestates, chargestates + 1):
        folder = 'charge_{}'.format(i)
        chargefile = folder + '/' + defect
        params = read_json(folder + '/params.json')
        q = params.get('charge')
        # elc = ElectrostaticCorrections(pristine=pristine_file,
        #                               charged=chargefile,
        #                               q=q,
        #                               sigma=sigma,
        #                               dimensionality=dim)
        print('INFO: using charged .gpw file "{}"'.format(chargefile))
        # elc.set_epsilons(epsilons)
        # eform = elc.calculate_corrected_formation_energy()
        # eform_array.append(eform)
        q_array.append(q)
    print(q_array)

    # ToDo: generate file with the results for eform and q_array which can
    #       can afterwards be read using 'postprocessing'

    return None


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
