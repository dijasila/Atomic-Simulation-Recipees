from asr.core import command, ASRResult, prepare_result  # , option
from pathlib import Path
from ase.io import Trajectory
from gpaw import restart
import typing


def webpanel(result, row, key_descriptions):
    from asr.database.browser import (fig, WebPanel,
                                      describe_entry, table, matrixtable)
    import numpy as np

    explained_keys = []
    for key in ['eform']:
        if key in result.key_descriptions:
            key_description = result.key_descriptions[key]
            explanation = key_description
            explained_key = describe_entry(key, description=explanation)
        else:
            explained_key = key
        explained_keys.append(explained_key)

    formation_table_sum = table(result, 'Defect properties', [])
    formation_table_sum['rows'].extend(
        [[describe_entry('Formation energy',
                         description=result.key_descriptions['eform']),
          f'{result.eform[0][0]:.2f} eV']])

    formation_table = table(result, 'Defect formation', [])
    for element in result.eform:
        formation_table['rows'].extend(
            [[describe_entry(f'Formation energy (q={element[1]:1d} @ VBM)',
                             description=result.key_descriptions['eform']),
              f'{element[0]:.2f} eV']])
    pristine_table_sum = table(result, 'Pristine crystal', [])
    pristine_table_sum['rows'].extend(
        [[describe_entry(f"Heat of formation",
                         description=result.key_descriptions['hof']),
          f"{result.hof:.2f} eV/atom"]])
    gap = result.pristine.cbm - result.pristine.vbm
    pristine_table_sum['rows'].extend(
        [[describe_entry("Band gap (PBE)",
                         description="Pristine band gap [eV]."),
          f"{gap:.2f} eV"]])

    trans_results = result.transitions
    vbm = result.pristine.vbm
    transition_labels = []
    transition_array = np.zeros((len(trans_results), 2))
    for i, element in enumerate(trans_results):
        transition_labels.append(element['transition_name'])
        transition_array[i, 0] = (element['transition_values']['transition']
                                  - element['transition_values']['evac']
                                  - vbm)
        transition_array[i, 1] = element['transition_values']['erelax']

    transitions_table = matrixtable(
        transition_array,
        title='Transition',
        columnlabels=[describe_entry('Transition Energy [eV]',
                                     description='SJ calculated transition level'),
                      describe_entry('Relaxation Correction [eV]',
                                     description='Correction due to ion relaxation')],
        rowlabels=transition_labels)

    panel = WebPanel(
        describe_entry('Charge Transition Levels (Slater-Janak)',
                       description='Defect stability analyzis using Slater-Janak theory'
                                   'to calculate charge transition levels and formation'
                                   'energies.'),
        columns=[[describe_entry(fig('sj_transitions.png'),
                                 'Slater-Janak calculated charge transition levels.'),
                  transitions_table],
                 [describe_entry(fig('formation.png'),
                                 'Reconstructed formation energy curve.'),
                  formation_table]],
        plot_descriptions=[{'function': plot_charge_transitions,
                            'filenames': ['sj_transitions.png']},
                           {'function': plot_formation_energies,
                            'filenames': ['formation.png']}],
        sort=50)

    summary = {'title': 'Summary',
               'columns': [[formation_table_sum,
                            pristine_table_sum],
                           []],
               'sort': 50}

    return [panel, summary]


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
class StandardStateResult(ASRResult):
    """Container for results related to the standard state of the present defect."""

    element: str
    eref: float

    key_descriptions = dict(
        element='Atomic species.',
        eref='Reference energy extracted from OQMD [eV].')


@prepare_result
class Result(ASRResult):
    """Container for Slater Janak results."""

    transitions: typing.List[TransitionResults]
    pristine: PristineResults
    eform: typing.List[typing.Tuple[float, int]]
    standard_states: typing.List[StandardStateResult]
    hof: float

    key_descriptions = dict(
        transitions='Charge transition levels with [transition energy, '
                    'relax correction, reference energy] eV',
        pristine='Container for pristine band gap results.',
        eform='List of formation energy tuples (eform wrt. standard states [eV], '
              'charge state)',
        standard_states='List of StandardStateResult objects for each species.',
        hof='Heat of formation for the pristine monolayer [eV]')

    formats = {"ase_webpanel": webpanel}


@command(module='asr.sj_analyze',
         requires=['sj_+0.5/gs.gpw', 'sj_-0.5/gs.gpw',
                   '../../unrelaxed.json',
                   'gs.gpw',
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
    print('INFO: calculate formation energy and charge transition levels '
          'for defect {}.'.format(defectsystem))

    # get heat of formation
    hof = get_heat_of_formation()

    # Obtain a list of all transitions with the respective ASRResults object
    transition_list = calculate_transitions()

    # get pristine band edges for correct referencing and plotting
    pris = get_pristine_band_edges()

    # get neutral formation energy without chemical potentials applied
    eform, standard_states = calculate_neutral_formation_energy()

    # get formation energies for all charge states based on neutral
    # formation energy, as well as charge transition levels, and pristine results
    eform = calculate_formation_energies(eform, transition_list, pris)

    return Result.fromdata(eform=eform,
                           transitions=transition_list,
                           pristine=pris,
                           standard_states=standard_states,
                           hof=hof)


def calculate_formation_energies(eform, transitions, pristine):
    """Calculate formation energies for all charge states at the VB band edge."""
    # from asr.core import read_json
    vbm = pristine['vbm']

    # CALCULATION OF FORMATION ENERGIES
    transitions = order_transitions(transitions)
    enlist = []
    for element in transitions:
        name = element['transition_name']
        q = name.split('/')[-1]
        if q < 0:
            enlist.append(element['transition_values']['transition']
                          - element['transition_values']['erelax']
                          - element['transition_values']['evac'])
        elif q > 0:
            enlist.append(element['transition_values']['transition']
                          + element['transition_values']['erelax']
                          - element['transition_values']['evac'])

    eform_list = [(eform, 0)]
    for i, element in enumerate(transitions):
        name = element['transition_name']
        q = name.split('/')[-1]
        if q < 0:
            enlist.append(element['transition_values']['transition']
                          - element['transition_values']['erelax']
                          - element['transition_values']['evac'])
        elif q > 0:
            enlist.append(element['transition_values']['transition']
                          + element['transition_values']['erelax']
                          - element['transition_values']['evac'])
        if name.split('/')[0].startswith('0') and name.split('/')[1].startswith('-'):
            y1 = eform
            y2 = eform
        elif name.split('/')[0].startswith('0'):
            y1 = eform
            y2 = eform
        elif not name.split('/')[1].startswith('-'):
            y2 = None
        else:
            y1 = None
        if name.split('/')[1].startswith('-'):
            a = float(name.split('/')[1])
            b = get_b(enlist[i], y2, a)
            if y1 is None:
                y1 = f(enlist[i], a, b)
            y2 = f(enlist[i + 1], a, b)
            eform_list.append((f(vbm, a, b), int(a)))
        elif not name.split('/')[0].startswith('-') or name.split('/').startswith('-'):
            a = float(name.split('/')[1])
            b = get_b(enlist[i], y1, a)
            if y2 is None:
                y2 = f(enlist[i], a, b)
            y1 = f(enlist[i + 1], a, b)
            eform_list.append((f(vbm, a, b), int(a)))

    return eform_list


def get_heat_of_formation():
    """Extract heat of formation from C2DB."""
    from asr.database.material_fingerprint import get_uid_of_atoms, get_hash_of_atoms
    from ase.db import connect
    from ase.io import read

    db = connect('/home/niflheim/fafb/db/c2db_july20.db')
    atoms = read('../../unrelaxed.json')
    hash = get_hash_of_atoms(atoms)
    uid = get_uid_of_atoms(atoms, hash)

    for row in db.select(uid=uid):
        hof = row.hform

    return hof


def get_kindlist():
    """Return list of elements present in the structure."""
    from ase.io import read

    atoms = read('structure.json')
    kindlist = []

    for symbol in atoms.get_chemical_symbols():
        if symbol not in kindlist:
            kindlist.append(symbol)

    return kindlist


def calculate_transitions():
    """Calculate all of the present transitions and return TransitionResults."""
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
    """Return band edges and vaccum level for the host system."""
    import numpy as np
    from asr.core import read_json

    print('INFO: extract pristine band edges.')
    p = Path('.')
    # sc = str(p.absolute()).split('/')[-2].split('_')[1].split('.')[0]
    # pristinelist = list(p.glob(f'./../../defects.pristine_sc.{sc}/'))
    # pris = pristinelist[0]
    pris = list(p.glob('./../../defects.pristine_sc*'))[0]
    if Path(pris / 'results-asr.gs.json').is_file():
        results_pris = read_json(pris / 'results-asr.gs.json')
        atoms, calc = restart('gs.gpw', txt=None)
        vbm = results_pris['vbm']
        cbm = results_pris['cbm']
        if not np.sum(atoms.get_pbc()) == 2:
            _, calc_pris = restart(pris / 'gs.gpw', txt=None)
            evac = calc_pris.get_eigenvalues()[0]
        else:
            evac = results_pris['evac']
    else:
        vbm = None
        cbm = None
        evac = None

    return PristineResults.fromdata(
        vbm=vbm,
        cbm=cbm,
        evac=evac)


def obtain_chemical_potential(symbol, db):
    """Extract the standard state of a given element."""
    energies_ss = []
    if symbol == 'v':
        eref = 0.
    else:
        for row in db.select(symbol, ns=1):
            energies_ss.append(row.energy / row.natoms)
        eref = min(energies_ss)

    return StandardStateResult.fromdata(
        element=symbol,
        eref=eref)


def get_defect_info():
    """Return defect_type, and defect_position of present defect."""
    from pathlib import Path

    p = Path('.')
    d_type = str(p.absolute()).split('/')[-2].split('_')[-2].split('.')[-1]
    d_pos = str(p.absolute()).split('/')[-2].split('_')[-1]

    return d_type, d_pos


def calculate_neutral_formation_energy():
    """Calculate the neutral formation energy without chemical potential shift applied.

    Only the neutral one is needed as for the higher charge states we will use the sj
    transitions for the formation energy plot.
    """
    from asr.core import read_json
    from ase.db import connect

    results_def = read_json('./results-asr.gs.json')
    p = Path('.')
    # sc = str(p.absolute()).split('/')[-2].split('_')[1].split('.')[0]
    # pristinelist = list(p.glob(f'./../../defects.pristine_sc.{sc}/'))
    # pris = pristinelist[0]
    pris = list(p.glob('./../../defects.pristine_sc*'))[0]
    results_pris = read_json(pris / 'results-asr.gs.json')

    eform = results_def['etot'] - results_pris['etot']

    # next, extract standard state energies for particular defect
    def_add, def_remove = get_defect_info()
    # extract standard states of defect atoms from OQMD
    db = connect('/home/niflheim/fafb/db/oqmd12.db')
    standard_states = []
    standard_states.append(obtain_chemical_potential(def_add, db))
    standard_states.append(obtain_chemical_potential(def_remove, db))

    eform = eform - standard_states[0].eref + standard_states[1].eref

    return eform, standard_states


def get_transition_level(transition, charge) -> TransitionResults:
    """Calculate the charge transition level for a given charge transition.

    :param transition: (List), transition (e.g. [0,-1])
    :param correct_relax: (Boolean), True if transition energy will be corrected
    """
    # import numpy as np
    # extract lowest lying state for the pristine system as energy reference
    p = Path('.')
    # sc = str(p.absolute()).split('/')[-2].split('_')[1].split('.')[0]
    # pris = Path(f'./../../defects.pristine_sc.{sc}/')
    pris = list(p.glob('./../../defects.pristine_sc*'))[0]
    _, calc_pris = restart(pris / 'gs.gpw', txt=None)
    e_ref_pris = calc_pris.get_eigenvalues()[0]

    # extrac HOMO or LUMO
    # HOMO
    charge = str(charge)
    if transition[0] > transition[1]:
        atoms, calc = restart('../charge_{}/sj_-0.5/gs.gpw'.format(charge), txt=None)
        # if not np.sum(atoms.get_pbc()) == 2:
        #     e_ref = calc.get_eigenvalues()[0]
        # else:
        #     e_ref_z = calc.get_electrostatic_potential().mean(0).mean(0)
        #     e_ref = (e_ref_z[0] + e_ref_z[-1]) / 2.
        ev = calc.get_eigenvalues()
        e_fermi = calc.get_fermi_level()
        unocc = []
        [unocc.append(v) for v in ev if v > e_fermi]
        e_trans = min(unocc)
        print('INFO: calculate transition level q = {} -> q = {} transition.'.format(
            transition[0], transition[1]))
    # LUMO
    elif transition[1] > transition[0]:
        atoms, calc = restart('../charge_{}/sj_+0.5/gs.gpw'.format(charge), txt=None)
        # if not np.sum(atoms.get_pbc()) == 2:
        #     e_ref = calc.get_eigenvalues()[0]
        # else:
        #     e_ref_z = calc.get_electrostatic_potential().mean(0).mean(0)
        #     e_ref = (e_ref_z[0] + e_ref_z[-1]) / 2.
        ev = calc.get_eigenvalues()
        e_fermi = calc.get_fermi_level()
        occ = []
        [occ.append(v) for v in ev if v < e_fermi]
        e_trans = max(occ)
        print('INFO: calculate transition level q = {} -> q = {} transition.'.format(
            transition[0], transition[1]))

    e_ref = calc.get_eigenvalues()[0] - e_ref_pris

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
    """Plot formation energies and transition levels within the gap."""
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.sj_analyze.json')

    vbm = data['pristine']['vbm']
    cbm = data['pristine']['cbm']
    gap = abs(cbm - vbm)
    eform = data['eform']
    transitions = data['transitions']

    fig, ax1 = plt.subplots()

    ax1.axvspan(-20, 0, color='grey', alpha=0.5)
    ax1.axvspan(gap, 20, color='grey', alpha=0.5)
    ax1.axhline(0, color='black', linestyle='dotted')
    for element in eform:
        ax1.plot([0, gap], [f(0, element[1], element[0]),
                            f(gap, element[1], element[0])],
                 #color='C0', 
                 label=element[1])
                 #linestyle='dotted')

    ax1.set_xlim(-0.2 * gap, gap + 0.2 * gap)
    # ax1.set_ylim(-0.1, eform[0][0] + 0.5 * eform[0][0])
    yrange = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    ax1.text(-0.1 * gap, 0.5 * yrange, 'VBM', ha='center',
             va='center', rotation=90, weight='bold', color='white')
    ax1.text(gap + 0.1 * gap, 0.5 * yrange, 'CBM', ha='center',
             va='center', rotation=90, weight='bold', color='white')

    tickslist = []
    labellist = []
    energies = []
    for i, element in enumerate(transitions):
        name = element['transition_name']
        q = name.split('/')[-1]
        if q < 0:
            energy = (element['transition_values']['transition']
                      - element['transition_values']['erelax']
                      - element['transition_values']['evac'] - vbm)
        elif q > 0:
            energy = (element['transition_values']['transition']
                      + element['transition_values']['erelax']
                      - element['transition_values']['evac'] - vbm)
        energies.append(energy)
        if energy > 0 and energy < (gap):
            tickslist.append(energy)
            labellist.append(name)
            ax1.axvline(energy, color='grey', linestyle='dotted')
    energies.append(100)

    # for i, element in enumerate(transitions):
    #     energy = energies[i]
    #     charge = int(element['transition_name'].split('/')[1])
    #     if energy > 0 and energy < gap:
    #         for en in eform:
    #             if en[1] == charge:
    #                 if charge < 0:
    #                     xmin = energy
    #                     xmax = min(gap, energies[i + 1])
    #                 else:
    #                     xmin = max(0, min(special, energies[i + 1]))
    #                     xmax = energy
    #                 # ax1.plot([xmin, xmax],
    #                 #          [f(xmin, en[1], en[0]),
    #                 #           f(xmax, en[1], en[0])],
    #                 #          color='black', linestyle='solid')
    # # ax1.plot([max(0, positive), min(gap, negative)], [eform[0][0], eform[0][0]],
    # #          color='black', linestyle='solid')

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([])
    # ax2.set_xticks(tickslist)
    # ax2.set_xticklabels(labellist)
    ax1.set_xlabel(r'$E - E_\mathrm{VBM}}$ [eV]')
    ax1.set_ylabel(r'$E^f$ (wrt. standard states) [eV]')
    ax1.legend()

    plt.tight_layout()

    plt.savefig(fname)
    plt.close()


def plot_charge_transitions(row, fname):
    """Plot calculated CTL along with the pristine bandgap."""
    import matplotlib.pyplot as plt

    data = row.data.get('results-asr.sj_analyze.json')

    vbm = data['pristine']['vbm']
    cbm = data['pristine']['cbm']

    gap = abs(cbm - vbm)

    transitions = data['transitions']

    # plt.plot([-2, 2], [vbm, vbm])
    # plt.plot([-2, 2], [cbm, cbm])

    plt.xlim(-1, 1)
    plt.ylim(-0.2 * gap, gap + 0.2 * gap)
    plt.xticks([], [])

    # plt.axhline(vbm, color='C0')
    # plt.axhline(cbm, color='C1')
    plt.axhspan(-5, 0, color='grey', alpha=0.5)
    plt.axhspan(gap, gap + 5, color='grey', alpha=0.5)
    plt.text(0, -0.1 * gap, 'VBM', color='white',
             ha='center', va='center', weight='bold')
    plt.text(0, gap + 0.1 * gap, 'CBM', color='white',
             ha='center', va='center', weight='bold')

    i = 1
    for trans in transitions:
        y = (trans['transition_values']['transition']
             - trans['transition_values']['erelax']
             - trans['transition_values']['evac'])
        if y <= (cbm + 0.2 * gap) and y >= (vbm - 0.2 * gap):
            plt.plot([-0.9, 0.5], [y - vbm, y - vbm], label=trans['transition_name'])
            # if i % 2 == 0:
            #     plt.text(0.6, y - vbm, trans['transition_name'], ha='left', va='center')
            # else:
            #     plt.text(-0.6, y - vbm,
            #              trans['transition_name'], ha='right', va='center')
            i += 1

    plt.legend(loc='center right')
    plt.ylabel(r'$E - E_{\mathrm{VBM}}$ [eV]')
    plt.yticks()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


if __name__ == '__main__':
    main.cli()
