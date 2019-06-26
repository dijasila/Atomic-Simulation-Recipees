import json
#from pathlib import Path
from asr.utils import command, option


@command('asr.defectformation')
@option('--pristine', default='../pristine/gs.gpw', 
        help='Relative path to ground state gpw file of pristine host system '
             'on which formation energy calculation is based. Here, the '
             'reference folder is the one with the defects and vacancies '
             'in it, as it was created from setup.defects.')
@option('--defect', default='gs.gpw',
        help='Ground state of disturbed system on which formation energy '
             'calculation is based')
@option('--size', default=None, help='Supercell size of both host and defect '
                                     'system')
#@option('--is2D/--is3D', default=True, help='Specify wheter you calculate '
                                    #        'the formation energy in 2D or '
                                     #       '3D')


def main(pristine, defect, size):
    """
    Calculate defect formation energy within a host crystal
    """
    import numpy as np
    from gpaw.defects import ElectrostaticCorrections
    from asr.utils import read_json
    from ase.io import read
    
    # ToDo: loop somehow over all charge states and return a list of 
    #       formation energies
    # ToDo: calculate sigma correctly for different systems

    # TBD!!!
    sigma = 1.0
    epsilons = [1.9, 1.15]

    # first, get supercell information from previous gs calculation
    
    # load relaxed pristine groundstate and relaxed impurity groundstate
    pristine_gs = pristine
    defect_gs = defect
    
    # get charge state of the defect system from params.json
    params = read_json('params.json')
    q = params.get('q') 

    # get dimensionality of the system
    #if is2D  == True:
    #    dim = '2d'
    #    #epsilons = [x, y]
    #elif is2D == False:
    #    dim = '3d'
    #    #epsilons = x

    # calculate electrostatic corrections of the charged defect
    elc = ElectrostaticCorrections(pristine=pristine_gs, 
                                   charged=defect_gs,
                                   q=q,
                                   sigma=sigma,
                                   dimensionality='2d')
    elc.set_epsilons(epsilons)

    # finally, compute the corrected formation energy
    eform = elc.calculate_corrected_formation_energy()

    print('formation energy of the defect: {}'.format(eform))

    return None


def collect_data():
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


group = 'property'
creates = []  # what files are created
dependencies = []  # no dependencies
resources = '1:10m'  # 1 core for 10 minutes
diskspace = 0  # how much diskspace is used
restart = 0  # how many times to restart

if __name__ == '__main__':
    main()

