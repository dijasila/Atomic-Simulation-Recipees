from asr.core import command, ASRResult, prepare_result  # , option
from pathlib import Path
from ase.io import Trajectory
from gpaw import restart
import typing


# TODO: fix sj-bug
# TODO: add chemical potential considerations
# TODO: add second y-axis
# TODO: implement reduced effective charge transition levels
# TODO: make webpanel compatible with new ASRResults object
# TODO: check whether new ASRResult object is correctly implemented

def webpanel(result, row, key_descriptions):
    from asr.database.browser import (fig, WebPanel, entry_parameter_description,
                                      describe_entry, table, matrixtable)
    import numpy as np

    # parameter_description = entry_parameter_description(
    #     row.data,
    #     'asr.sj_analyze')
    # print(parameter_description)

    # explained_keys = []
    # for key in ['transitions', 'eform', 'pristine']:
    #     if key in result.description:
    #         key_description = result.key_descriptions[key]
    #         explanation = (f'{key_description} '
    #                         'blablabla.\n\n'
    #                         + parameter_description)
    #         explained_key = describe_entry(key, description=explanation)
    #     else:
    #         explained_key = key
    #     explained_keys.append(explained_key)

    trans_results = result.transitions
    transition_labels = []
    transition_array = np.zeros((len(trans_results), 3))
    for i, element in enumerate(trans_results):
        transition_labels.append(element['transition_name'])
        transition_array[i, 0] = element['transition_values']['transition']
        transition_array[i, 1] = element['transition_values']['erelax']
        transition_array[i, 2] = element['transition_values']['evac']
    # transitions_table = table(trans_results[0], 'Transition levels',
    #                           ['transition_values', trans_results[0]['transition_values']['transition'],
    #                            trans_results[0]['transition_values']['erelax']],
    #                           key_descriptions, 4)

    transitions_table = matrixtable(transition_array,
                                    title='Transition Levels [eV]',
                                    columnlabels=['Transition', 'Relax contribution', 'Vacuum level'],
                                    rowlabels=transition_labels)

    panel = WebPanel('Charge Transition Levels (Slater-Janak)',
                     columns=[[transitions_table], [describe_entry(fig('sj_transitions.png'), 'transitions')],
                              [describe_entry(fig('sj_transsitions.png'), '2transitions')]],
                     plot_descriptions=[{'function': plot_charge_transitions,
                                         'filenames': ['sj_transitions.png']}
                                         ],
                     sort=11)

    formation = WebPanel('Defect Stability',
                         columns=[[describe_entry(fig('formation.png'), 'Formation energies')]],
                         plot_descriptions=[{'function': plot_formation_energies,
                                             'filenames': ['formation.png']},
                                             ],
                         sort=12)

    # summary = WebPanel(title=describe_entry('Summary',
    #     description='This panel contains a summary of the most '
    #                 'important properties of the material.'),
    #     columns=[],
    #     plot_descriptions=[{'function': plot_charge_transitions,
    #                         'filenames': ['sj_transitions.png']}],
    #     sort=10)

    return [panel, formation]


@prepare_result
class PristineResults(ASRResult):
    """Container for pristine band gap results."""
    vbm: float
    cbm: float
    evac: float

    key_descriptions = dict(
        vbm='Pristine valence band maximum [eV].',
        cbm='Pristien conduction band minimum [eV]',
        evac='Pristine vacuum level [eV]')


@prepare_result
class TransitionValues(ASRResult):
    """Container for values of a specific charge transition level."""
    transition: float
    erelax: float
    evac: float

    key_descriptions = dict(
        transition='Charge transition level [eV]',
        erelax='Reorganization contribution  to the transition level [eV]',
        evac='Vacuum level for halfinteger calculation [eV]')


@prepare_result
class TransitionResults(ASRResult):
    """Container for charge transition level results."""
    transition_name: str
    transition_values: TransitionValues

    key_descriptions = dict(
        transition_name='Name of the charge transition (Initial State/Final State)',
        transition_values='Container for values of a specific charge transition level.')


@prepare_result
class TransitionListResults(ASRResult):
    """Container for all charge transition level results."""
    transition_list: typing.List[TransitionResults]

    key_descriptions = dict(
        transition_list='List of TransitionResults objects.')


@prepare_result
class Result(ASRResult):
    """Container for Slater Janak results."""
    transitions: typing.List[TransitionResults]
    pristine: PristineResults
    eform: float

    key_descriptions = dict(
        transitions='Charge transition levels with [transition energy, '
                        'relax correction, reference energy] eV',
        pristine='Container for pristine band gap results.',
        eform='Neutral formation energy without chemical potentials applied [eV]')

    formats = {"ase_webpanel": webpanel}


@command(module='asr.sj_analyze',
         webpanel=webpanel,
         requires=['sj_+0.5/gs.gpw', 'sj_-0.5/gs.gpw',
                   'results-asr.setup.defects.json'],
         resources='24:2h',
         returns=Result)
def main() -> Result:
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

    # Obtain a list of all transitions with the respective ASRResults object
    transition_list = calculate_transitions()

    # get pristine band edges for correct referencing and plotting
    pris = get_pristine_band_edges()

    # get neutral formation energy without chemical potentials applied
    eform = calculate_neutral_formation_energy()

    return Result.fromdata(transitions=transition_list,
                           pristine=pris,
                           eform=eform)


def calculate_transitions():
    """Calculate all of the present transitions and return an
    ASRResults object (TransitionResults)."""

    transition_list = []
    # First, get IP and EA (charge transition levels for the neutral defect
    if Path('./sj_+0.5/gs.gpw').is_file() and Path('./sj_-0.5/gs.gpw').is_file():
        transition = [0, +1]
        transition_results = get_transition_level(transition, 0)
        transition_list.append(transition_results)
        transition = [0, -1]
        transition_results = get_transition_level(transition, 0)
        transition_list.append(transition_results)

    for q in [-3, -2, -1, 1, 2, 3]:
        if q > 0 and Path('./../charge_{}/sj_+0.5/gs.gpw'.format(q)).is_file():
            transition = [q, q + 1]
            transition_results = get_transition_level(transition, q)
            transition_list.append(transition_results)
        if q < 0 and Path('./../charge_{}/sj_-0.5/gs.gpw'.format(q)).is_file():
            transition = [q, q - 1]
            transition_results = get_transition_level(transition, q)
            transition_list.append(transition_results)

    return transition_list


def get_pristine_band_edges() -> PristineResults:
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
        # evac = results_pris['evac']
        evac_z = calc.get_electrostatic_potential().mean(0).mean(0)
        evac = (evac_z[0] + evac_z[-1]) / 2.
    else:
        vbm = None
        cbm = None
        evac = None

    return PristineResults.fromdata(
        vbm=vbm,
        cbm=cbm,
        evac=evac)


def obtain_chemical_potential():
    """
    Function to evaluate the chemical potential limits for a given defect.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TBD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    return None


def calculate_neutral_formation_energy():
    """
    Function to calculate the neutral formation energy without chemical
    potential shift applied. Only the neutral one is needed as for the higher
    charge states we will use the sj transitions for the formation energy
    plot.
    """
    from asr.core import read_json
    results_def = read_json('./results-asr.gs.json')
    results_pris = read_json('./../../defects.pristine_sc/results-asr.gs.json')

    eform = results_def['etot'] - results_pris['etot']

    return eform


def get_transition_level(transition, charge) -> TransitionResults:
    """
    Calculates the charge transition level for a given charge transition.

    :param transition: (List), transition (e.g. [0,-1])
    :param correct_relax: (Boolean), True if transition energy will be corrected
    """
    # extrac HOMO or LUMO
    # HOMO
    charge = str(charge)
    if transition[0] > transition[1]:
        _, calc = restart('../charge_{}/sj_-0.5/gs.gpw'.format(charge), txt=None)
        e_ref_z = calc.get_electrostatic_potential().mean(0).mean(0)
        e_ref = (e_ref_z[0] + e_ref_z[-1]) / 2.
        ev = calc.get_eigenvalues()
        e_fermi = calc.get_fermi_level()
        unocc = []
        [unocc.append(v) for v in ev if v > e_fermi]
        e_trans = min(unocc)
        print('INFO: calculate transition level q = {} -> q = {} transition.'.format(
            transition[0], transition[1]))
    # LUMO
    elif transition[1] > transition[0]:
        _, calc = restart('../charge_{}/sj_+0.5/gs.gpw'.format(charge), txt=None)
        e_ref_z = calc.get_electrostatic_potential().mean(0).mean(0)
        e_ref = (e_ref_z[0] + e_ref_z[-1]) / 2.
        ev = calc.get_eigenvalues()
        e_fermi = calc.get_fermi_level()
        occ = []
        [occ.append(v) for v in ev if v < e_fermi]
        e_trans = max(occ)
        print('INFO: calculate transition level q = {} -> q = {} transition.'.format(
            transition[0], transition[1]))

    # if possible, calculate correction due to relaxation in the charge state
    if Path('../charge_{}/results-asr.relax.json'.format(
            int(transition[1]))).is_file():
        print('INFO: calculate relaxation contribution to transition level.')
        traj = Trajectory('../charge_{}/relax.traj'.format(str(int(transition[1]))))
        e_cor = traj[0].get_potential_energy() - traj[-1].get_potential_energy()
    else:
        print('INFO: no relaxation for the charged state present. Do not calculate '
              'relaxation contribution to transition level.')
        e_cor = 0

    transition_name = f'{transition[0]}/{transition[1]}'

    transition_values = return_transition_values(e_trans, e_cor, e_ref)

    return TransitionResults.fromdata(
        transition_name=transition_name,
        transition_values=transition_values)


def return_transition_values(e_trans, e_cor, e_ref) -> TransitionValues:
    return TransitionValues.fromdata(
        transition=e_trans,
        erelax=e_cor,
        evac=e_ref)


def order_transitions(transitions):
    translist = []
    nameslist = []
    reflist = ['0/1', '1/2', '2/3', '0/-1', '-1/-2', '-2/-3']

    for element in transitions:
        if element['transition_name'] in reflist:
            translist.append(element)
            nameslist.append(element['transition_name'])

    ordered_list = []
    for element in reflist:
        for oelement in transitions:
            if element == oelement['transition_name']:
                ordered_list.append(oelement)

    return ordered_list


def get_b(x, y, a):
    return y - a * x


def f(x, a, b):
    return a * x + b


def plot_formation_energies(row, fname):
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.sj_analyze.json')

    vbm = data['pristine']['vbm'] - data['pristine']['evac']
    cbm = data['pristine']['cbm'] - data['pristine']['evac']
    gap = cbm - vbm
    eform = data['eform']

    transitions = data['transitions']

    fig, ax1 = plt.subplots()

    ax1.fill_betweenx([-10, 30], vbm - 10, vbm, color='C0', alpha=0.5)
    ax1.fill_betweenx([-10, 30], cbm + 10, cbm, color='C1', alpha=0.5)
    ax1.axvline(vbm, color='C0')
    ax1.axvline(cbm, color='C1')
    ax1.axhline(0, color='black', linestyle='dotted')
    ax1.plot([vbm, cbm], [eform, eform], color='black')

    ax1.set_xlim(vbm - 0.2 * gap, cbm + 0.2 * gap)
    ax1.set_ylim(-0.1, eform + 0.2 * eform)
    yrange = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    ax1.text(vbm - 0.1 * gap, 0.5 * yrange, 'VBM', ha='center', va='center', rotation=90, weight='bold', color='white')
    ax1.text(cbm + 0.1 * gap, 0.5 * yrange, 'CBM', ha='center', va='center', rotation=90, weight='bold', color='white')

    for trans in transitions:
        if trans['transition_name'] == '0/1' or trans['transition_name'] == '0/-1':
            trans_val = trans['transition_values']
            e_pm = trans_val['transition'] - trans_val['erelax'] - trans_val['evac']
        if e_pm < cbm and e_pm > vbm:
            ax1.axvline(e_pm, color='black', linestyle='-.')

    transitions = order_transitions(transitions)

    enlist = []
    for element in transitions:
        enlist.append(element['transition_values']['transition'] -
                      element['transition_values']['erelax'] -
                      element['transition_values']['evac'])

    ax2 = ax1.twiny()

    tickslist = []
    labellist = []
    for i, element in enumerate(transitions):
        energy = (element['transition_values']['transition'] -
                  element['transition_values']['erelax'] -
                  element['transition_values']['evac'])
        enlist.append(energy)
        name = element['transition_name']
        if energy > vbm and energy < cbm:
            ax1.axvline(energy, color='grey', linestyle='-.')
            if name.split('/')[0].startswith('0') and name.split('/')[1].startswith('-'):
                y1 = eform
                y2 = eform
            elif name.split('/')[1].startswith('0'):
                y1 = eform
                y2 = eform
            elif not name.split('/')[1].startswith('-'):
                y2 = None
            else:
                y1 = None
            if name.split('/')[1].startswith('-'):
                tickslist.append(energy)
                labellist.append(name)
                a = float(name.split('/')[1])
                b = get_b(enlist[i], y2, a)
                if y1 is None:
                    y1 = f(enlist[i], a, b)
                y2 = f(enlist[i + 1], a, b)
                print(enlist[i], enlist[i+1], y1, y2, a)
                ax1.plot([vbm, cbm], [f(vbm, a, b), f(cbm, a, b)], color='black')
                ax1.plot([enlist[i], enlist[i + 1]], [y1, y2], color='black', marker='s')
            elif not name.split('/')[0].startswith('-') or name.split('/').startswith('-'):
                tickslist.append(energy)
                labellist.append(name)
                a = float(name.split('/')[1])
                b = get_b(enlist[i], y1, a)
                if y2 is None:
                    y2 = f(enlist[i], a, b)
                y1 = f(enlist[i + 1], a, b)
                print(enlist[i], enlist[i+1], y1, y2, a)
                ax1.plot([vbm, cbm], [f(vbm, a, b), f(cbm, a, b)], color='black')
                ax1.plot([enlist[i + 1], enlist[i]], [y1, y2], color='black', marker='s')
    ax1.set_xlabel('$E_F$ [eV]')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(tickslist)
    ax2.set_xticklabels(labellist)

    ax1.set_ylabel('Formation energy [eV]')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_charge_transitions(row, fname):
    """
    Plot the calculated charge transition levels along with the pristine bandgap.
    """
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.sj_analyze.json')

    vbm = data['pristine']['vbm'] - data['pristine']['evac']
    cbm = data['pristine']['cbm'] - data['pristine']['evac']
    gap = cbm - vbm

    transitions = data['transitions']

    plt.plot([-2, 2], [vbm, vbm])
    plt.plot([-2, 2], [cbm, cbm])

    plt.xlim(-1, 1)
    plt.ylim(vbm - 0.2 * gap, cbm + 0.2 * gap)
    plt.xticks([], [])

    plt.axhline(vbm, color='C0')
    plt.axhline(cbm, color='C1')
    plt.fill_between([-2, 2], [vbm, vbm], [vbm - 2, vbm - 2], color='C0', alpha=0.5)
    plt.fill_between([-2, 2], [cbm, cbm], [0, 0], color='C1', alpha=0.5)
    plt.text(0, vbm - 0.1 * gap, 'VBM', color='white', ha='center', va='center', weight='bold')
    plt.text(0, cbm + 0.1 * gap, 'CBM', color='white', ha='center', va='center', weight='bold')

    i = 1
    for trans in transitions:
        y = (trans['transition_values']['transition'] -
             trans['transition_values']['erelax'] -
             trans['transition_values']['evac'])
        plt.plot([-0.5, 0.5], [y, y], color='black')
        if i % 2 == 0:
            plt.text(0.6, y, trans['transition_name'], ha='left', va='center')
        else:
            plt.text(-0.6, y, trans['transition_name'], ha='right', va='center')
        i += 1

    plt.ylabel('$E - E_{vac}$ [eV]')
    plt.yticks()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    main.cli()
