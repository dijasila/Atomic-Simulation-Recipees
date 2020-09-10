from asr.core import command, option
from gpaw import restart


@command(module='asr.analyze_state',
         requires=['gs.gpw'],
         resources='1:1h')
@option('--spin', help='Specify which spin channel you want to consider. '
        'Choose 0 for the first spin channel, 1 for the second spin channel, '
        'and 2 if both should be considered.', type=int)
@option('--state', help='Specify the specific state (band number) that you '
        'want to consider. Note, that this argument is not used when the '
        'gap state flag is active.', type=int)
@option('--get-gapstates/--dont-get-gapstates', help='Should all of the gap'
        ' states be saved and analyzed? Note, that if gap states are analysed'
        ' the --state option will be neglected.', is_flag=True)
def main(spin: int = 2, state: int = 0, get_gapstates: bool = False):
    """Write out wavefunction and analyze it.

    This recipe reads in an existing gs.gpw file and writes out wavefunctions
    of different states (either the one of a specific given bandindex or of
    all the defect states in the gap). Furthermore, it will feature some post
    analysis on those states.
    """

    atoms, calc = restart('gs.gpw', txt=None)
    if spin == 0 or spin == 2:
        states_0 = return_gapstates(calc, spin=0)
    elif spin == 1 or spin == 2:
        states_1 = return_gapstates(calc, spin=1)

    print(states_0, states_1)

    return None


def return_gapstates(calc_def, spin=0):
    """Evaluates which states are inside the gap and returns the band indices
    of those states for a given spin channel.
    """
    from asr.core import read_json

    _, calc_pris = restart('../../defects.pristine_sc/gs.gpw')
    results_pris = read_json('../../defects.pristine_sc/results-asr.gs.json')
    results_def = read_json('results-asr.gs.json')
    vbm = results_pris['vbm'] - results_pris['evac']
    cbm = results_pris['cbm'] - results_pris['evac']

    es_def = calc_def.get_eigenvalues() - results_def['evac']
    es_pris = calc_pris.get_eigenvalues() - results_pris['evac']

    diff = es_pris[0] - es_def[0]
    states_def = es_def + diff

    statelist = []
    [statelist.append(i) for state, i in enumerate(states_def) if (
        state < cbm and state > vbm)]
    # for state, i in enumerate(states_def):
    #     if state < cbm and state > vbm:
    #         statelist.append(i)

    return stateist
