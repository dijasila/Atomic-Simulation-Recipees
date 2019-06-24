import json
#from pathlib import Path
from asr.utils import command, option


@command('asr.defectformation')
@option('--hostfile', default='../pristine/gs.gpw', 
        help='Relative path to ground state gpw file of pristine host system '
             'on which formation energy calculation is based. Here, the '
             'reference folder is the one with the defects and vacancies '
             'in it, as it was created from setup.defects.')
@option('--defectfile', default='gs.gpw',
        help='Ground state of disturbed system on which formation energy '
             'calculation is based')
@option('--size', default=None, help='Supercell size of both host and defect '
                                     'system')
@option('--is2D/--is3D', default=True, help='Specify wheter you calculate '
                                            'the formation energy in 2D or '
                                            '3D')


def main():
    """
    Calculate defect formation energy within a host crystal
    """
    import numpy as np
    from gpaw.defects import ElectrostaticCorrections
    from ase.io import read
    
    # first, get supercell information from previous gs calculation
    

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

