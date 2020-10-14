from asr.core import command, option
from pathlib import Path
from ase.io import Trajectory
from gpaw import restart


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

    # Initialize results dictionary
    results = {}

    # TEMPORARY PART!!!
    correct_relax = False

    # First, get IP and EA (charge transition levels for the neutral defect
    if Path('./sj_+0.5/gs.gpw').is_file() and Path('./sj_-0.5/gs.gpw').is_file():
        transition = [0, +1]
        e_trans, e_cor, e_ref = get_transition_level(transition, correct_relax)
        results['{}/{}'.format(transition[0], transition[1])] = [
                e_trans, e_cor, e_ref]
        transition = [0, -1]
        e_trans, e_cor, e_ref = get_transition_level(transition, correct_relax)
        results['{}/{}'.format(transition[1], transition[0])] = [
                e_trans, e_cor, e_ref]

    for q in [-3, -2, -1, 1, 2, 3]:
        if q > 0 and Path('./../charge_{}/sj_+0.5/gs.gpw'.format(q)).is_file():
            transition = [q, q + 1]
            e_trans, e_cor, e_ref = get_transition_level(transition, correct_relax)
            results['{}/{}'.format(transition[0], transition[1])] = [
                    e_trans, e_cor, e_ref]
        if q < 0 and Path('./../charge_{}/sj_-0.5/gs.gpw'.format(q)).is_file():
            transition = [q, q - 1]
            e_trans, e_cor, e_ref = get_transition_level(transition, correct_relax)
            results['{}/{}'.format(transition[0], transition[1])] = [
                    e_trans, e_cor, e_ref]


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

    # extrac HOMO or LUMO
    # HOMO
    if transition[0] > transition[1]:
        calc = restart('sj_-0.5/gs.gpw', txt=None)
        e_ref = calc.get_electrostatic_potential()[0,0,0]
        ev = calc.get_eigenvalues()
        e_fermi = calc.get_fermi_level()
        occ = []
        [occ.append(v) for v in ev if v < e_fermi]
        e_trans = max(occ)
    # LUMO
    elif transition[1] > transition[0]:
        calc = restart('sj_+0.5/gs.gpw', txt=None)
        e_ref = calc.get_electrostatic_potential()[0,0,0]
        ev = calc.get_eigenvalues()
        e_fermi = calc.get_fermi_level()
        unocc = []
        [unocc.append(v) for v in ev if v > e_fermi]
        e_trans = min(unocc)

    return e_trans, e_cor, e_ref


if __name__ == '__main__':
    main.cli()
