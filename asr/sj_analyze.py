from asr.core import command, option
from pathlib import Path
from ase.io import Trajectory


@command(module='asr.sj_analyze',
         requires=['sj_+0.5/gs.gpw', 'sj_-0.5/gs.gpw',
                   'results-asr.setup.defects.json'],
         resources='24:2h')
def main():
    """Calculate charge transition levels for defect systems.

    This recipe uses SJ theory to calculate charge transition levels for defect systems.
    At least, asr.setup.sj had to be run in the charge_0 folder of a defect system and
    the half integer calculations have to be finished within the newly created folders.
    """
    p = Path('.')
    defectsystem = str(p.absolute()).split('/')[-2]
    print('INFO: calculate charge transition levels for defect {}.'.format(
        defectsystem))
    results = {}

    return results


def get_transition_level(transition, correct_relax):
    """
    Calculates the charge transition level for a given charge transition.

    :param transition: (List), transition (e.g. [0,-1])
    :param correct_relax: (Boolean), True if transition energy will be corrected
    """
    # if possible, calculate correction due to relaxation in the charge state
    if correct_relax:
        traj = Trajectory('../charge_{}/relax.traj'.format(str(int(transition[1]))))
        e_cor = traj[0].get_potential_energy() - traj[-1].get_potential_energy()
    else:
        e_cor = 0

    if transition[0] > transition[1]:
        calc = restart('sj_-0.5/gs.gpw', txt=None)
    elif transition[1] > transition[0]:
        calc = restart('sj_+0.5/gs.gpw', txt=None)

    # extract HOMO or LUMO
    # TODO: TBD

    # reference to vaccum level
    # TODO: TBD

    # subtract relaxation correction

    # return e_trans, e_cor, e_ref


if __name__ == '__main__':
    main.cli()
