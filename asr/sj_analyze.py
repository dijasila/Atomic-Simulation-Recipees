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
    from asr.database.browser import fig

    panel = {'title': 'Charge transition levels and pristine band edges',
             'columns': [fig('sj_transitions.png')],
             'plot_descriptions': [{'function': plot_charge_transitions,
                                    'filenames': ['sj_transitions.png']}],
             'sort': 12}

    return [panel]


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
        transition_results = get_transition_level(transition)
        transition_list.append(transition_results)
        transition = [0, -1]
        transition_results = get_transition_level(transition)
        transition_list.append(transition_results)

    # for q in [-3, -2, -1, 1, 2, 3]:
    #     if q > 0 and Path('./../charge_{}/sj_+0.5/gs.gpw'.format(q)).is_file():
    #         transition = [q, q + 1]
    #         transition_results = get_transition_level(transition)
    #         transition_list.append(transition_results)
    #     if q < 0 and Path('./../charge_{}/sj_-0.5/gs.gpw'.format(q)).is_file():
    #         transition = [q, q - 1]
    #         transition_results = get_transition_level(transition)
    #         transition_list.append(transition_results)

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


def get_transition_level(transition) -> TransitionResults:
    """
    Calculates the charge transition level for a given charge transition.

    :param transition: (List), transition (e.g. [0,-1])
    :param correct_relax: (Boolean), True if transition energy will be corrected
    """
    # extrac HOMO or LUMO
    # HOMO
    if transition[0] > transition[1]:
        _, calc = restart('sj_-0.5/gs.gpw', txt=None)
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
        _, calc = restart('sj_+0.5/gs.gpw', txt=None)
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


# def plot_formation_energies(row, fname):
#     """
#     Using the calculated charge transition levels, plot the formation energy curve
#     for a given defect as a function of the fermi energy.
#     """
#     import matplotlib.pyplot as plt
# 
#     data = row.data.get('results-asr.sj_analyze.json')
#     eform = data['eform']
# 
#     vbm = data['pristine']['vbm'] - data['pristine']['evac']
#     cbm = data['pristine']['cbm'] - data['pristine']['evac']
# 
#     transitions = data['transitions']
# 
#     fig, ax1 = plt.subplots()
# 
#     ax1.fill_betweenx([-10, 30], vbm - 10, vbm, color='C0', alpha=0.5)
#     ax1.fill_betweenx([-10, 30], cbm + 10, cbm, color='C1', alpha=0.5)
#     ax1.axhline(0, color='black', linestyle='dotted')
# 
#     plt.xlim(vbm - (0.1 * (cbm - vbm)), cbm + (0.1 * (cbm - vbm)))
#     plt.ylim(-1, eform + 0.1 * eform)
#     energy_m = transitions["-1/0"][0] - transitions["-1/0"][1] - transitions["-1/0"][2]
#     energy_p = transitions["0/1"][0] - transitions["0/1"][1] - transitions["0/1"][2]
#     ax1.plot([max(energy_p, vbm), min(energy_m, cbm)], [eform, eform], color='black')
# 
#     translist_m = []
#     translist_p = []
#     for i, element in enumerate(sorted(transitions)):
#         energy = transitions[element][0] - transitions[element][1] - transitions[element][2]
#         name = element
#         if name.split('/')[0].startswith('-'):
#             translist_m.append(energy)
#         else:
#             translist_p.append(energy)
#     translist_m.append(cbm)
#     translist_p.append(vbm)
# 
#     ax2 = ax1.twiny()
# 
#     i = 0
#     j = 0
#     enlist = []
#     tickslist = []
#     for element in sorted(transitions):
#         energy = transitions[element][0] - transitions[element][1] - transitions[element][2]
#         name = element
#         if name.split('/')[1].startswith('0') and name.split('/')[0].startswith('-'):
#             y_1 = eform
#             y_2 = eform
#         else:
#             y_1 = None
#         if name.split('/')[0].startswith('-'):
#             enlist.append(energy)
#             tickslist.append(name)
#             a = float(name.split('/')[0])
#             b = y_2 - a * translist_m[i]
#             if y_1 is None:
#                 y_1 = a * translist_m[i] + b
#             y_2 = a * translist_m[i + 1] + b
#             ax1.plot([energy, translist_m[i + 1]], [y_1, y_2], color='black')
#             i += 1
#         ax1.axvline(energy, color='grey', linestyle='-.')
#     for element in sorted(transitions):
#         energy = transitions[element][0] - transitions[element][1] - transitions[element][2]
#         name = element
#         if name.split('/')[0].startswith('0') and not name.split('/')[0].startswith('-'):
#             y_1 = eform
#             y_2 = eform
#         else:
#             y_2 = None
#         if not name.split('/')[0].startswith('-'):
#             enlist.append(energy)
#             tickslist.append(name)
#             a = float(name.split('/')[1])
#             print(a)
#             b = y_1 - a * translist_p[j]
#             diff = abs(translist_p[j] - translist_p[j + 1])
#             if y_2 is None:
#                 y_2 = a * translist_p[j] + b
#             y_1 = a * (translist_p[j] - diff) + b
#             print(y_1, y_2, translist_p[j], translist_p[j + 1])
#             ax1.plot([translist_p[j + 1], energy], [y_1, y_2], color='black')
#             j += 1
#         ax1.axvline(energy, color='grey', linestyle='-.')
# 
#     ax1.set_ylabel('$E_{form}$ (eV)')
#     ax1.set_xlabel('$E_F$ (eV)')
#     ax2.set_xlim(ax1.get_xlim())
#     ax2.set_xticks(enlist)
#     ax2.set_xticklabels(tickslist)
#     plt.tight_layout()
#     plt.savefig(fname)
#     plt.close()


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
