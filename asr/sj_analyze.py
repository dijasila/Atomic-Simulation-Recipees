from asr.core import command  # , option
from pathlib import Path
from ase.io import Trajectory
from gpaw import restart


def webpanel(row, key_descriptions):
    from asr.database.browser import fig

    panel = {'title': 'Charge transition levels and pristine band edges',
             'columns': [fig('sj_transitions.png')],
             'plot_descriptions': [{'function': plot_charge_transitions,
                                    'filenames': ['sj_transitions.png']}],
             'sort': 12}

    return [panel]


@command(module='asr.sj_analyze',
         webpanel=webpanel,
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

    vbm, cbm, evac = get_pristine_band_edges()
    results['pristine'] = {'vbm': vbm, 'cbm': cbm, 'evac': evac}

    return results


def get_pristine_band_edges():
    """
    Returns band edges and vaccum level for the host system.
    """
    from asr.core import read_json

    print('INFO: extract pristine band edges.')
    if Path('./../../defects.pristine_sc/results-asr.gs.json').is_file():
        results_pris = read_json('./../../defects.pristine_sc/results-asr.gs.json')
        _, calc = restart('gs.gpw', txt=None)
        vbm = results_pris['vbm']
        cbm = results_pris['cbm']
        evac = calc.get_electrostatic_potential()[0, 0, 0]
        # evac = results_pris['evac']
    else:
        vbm = None
        cbm = None
        evac = None

    return vbm, cbm, evac


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
        _, calc = restart('sj_-0.5/gs.gpw', txt=None)
        e_ref = calc.get_electrostatic_potential()[0, 0, 0]
        ev = calc.get_eigenvalues()
        e_fermi = calc.get_fermi_level()
        occ = []
        [occ.append(v) for v in ev if v < e_fermi]
        e_trans = max(occ)
        print('INFO: calculate transition level q = {} -> q = {} transition.'.format(
            transition[0], transition[1]))
    # LUMO
    elif transition[1] > transition[0]:
        _, calc = restart('sj_+0.5/gs.gpw', txt=None)
        e_ref = calc.get_electrostatic_potential()[0, 0, 0]
        ev = calc.get_eigenvalues()
        e_fermi = calc.get_fermi_level()
        unocc = []
        [unocc.append(v) for v in ev if v > e_fermi]
        e_trans = min(unocc)
        print('INFO: calculate transition level q = {} -> q = {} transition.'.format(
            transition[0], transition[1]))

    return e_trans, e_cor, e_ref


def plot_charge_transitions(row, fname):
    """
    Plot the calculated charge transition levels along with the pristine bandgap.
    """
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.sj_analyze.json')

    vbm = data['pristine']['vbm'] - data['pristine']['evac']
    cbm = data['pristine']['cbm'] - data['pristine']['evac']

    transitions = data['transitions']

    plt.plot([-2, 2], [vbm, vbm])
    plt.plot([-2, 2], [cbm, cbm])

    plt.xlim(-1, 1)
    plt.ylim(vbm - 1, cbm + 1)
    plt.xticks([], [])

    plt.fill_between([-2, 2], [vbm, vbm], [vbm - 2, vbm - 2], color='C0')
    plt.fill_between([-2, 2], [cbm, cbm], [0, 0], color='C1')

    i = 1
    for t in transitions:
        y = transitions[t][0] - transitions[t][2]
        plt.plot([-0.5, 0.5], [y, y], color='black')
        if i % 2 == 0:
            plt.text(0.6, y, t, fontsize=14)
        else:
            plt.text(-0.7, y, t, fontsize=14)
        i += 1

    plt.ylabel('$E - E_{vac}$ in eV', fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    main.cli()
