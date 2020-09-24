from ase.io import read, write
from asr.core import command, option
from gpaw import GPAW, restart


@command(module='asr.analyze_state',
         requires=['gs.gpw', 'structure.json',
                   '../../defects.pristine_sc/gs.gpw'],
         resources='12:1h')
@option('--state', help='Specify the specific state (band number) that you '
        'want to consider. Note, that this argument is not used when the '
        'gap state flag is active.', type=int)
@option('--get-gapstates/--dont-get-gapstates', help='Should all of the gap'
        ' states be saved and analyzed? Note, that if gap states are analysed'
        ' the --state option will be neglected.', is_flag=True)
@option('--analyze/--dont-analyze', help='Not only create cube files of '
        'specific states, but also analyze them.', is_flag=True)
def main(state: int = 0,
         get_gapstates: bool = False,
         analyze: bool = False):
    """Write out wavefunction and analyze it.

    This recipe reads in an existing gs.gpw file and writes out wavefunctions
    of different states (either the one of a specific given bandindex or of
    all the defect states in the gap). Furthermore, it will feature some post
    analysis on those states.

    Test.
    """
    atoms = read('structure.json')
    print('INFO: run fixdensity calculation')
    calc = GPAW('gs.gpw', txt='analyze_states.txt')
    calc = calc.fixed_density(kpts={'size': (1, 1, 1), 'gamma': True})
    calc.get_potential_energy()
    if get_gapstates:
            print('INFO: evaluate gapstates ...')
            states = return_gapstates_fix(calc, spin=0)
    elif not get_gapstates:
        states = [state]

    print('INFO: write wavefunctions of gapstates ...')
    for band in states:
        wf = calc.get_pseudo_wave_function(band=band, spin=0)
        fname = 'wf.{0}_{1}.cube'.format(band, 0)
        write(fname, atoms, data=wf)
        if calc.get_number_of_spins() == 2:
            wf = calc.get_pseudo_wave_function(band=band, spin=1)
            fname = 'wf.{0}_{1}.cube'.format(band, 1)
            write(fname, atoms, data=wf)

    if analyze:
        # To be implemented
        print('INFO: analyze chosen states.')

    return None


def return_gapstates(calc_def, spin=0):
    """Evaluates which states are inside the gap and returns the band indices
    of those states for a given spin channel.
    """
    from asr.core import read_json

    _, calc_pris = restart('../../defects.pristine_sc/gs.gpw', txt=None)
    results_pris = read_json('../../defects.pristine_sc/results-asr.gs.json')
    results_def = read_json('results-asr.gs.json')
    vbm = results_pris['vbm'] - results_pris['evac']
    cbm = results_pris['cbm'] - results_pris['evac']

    es_def = calc_def.get_eigenvalues() - results_def['evac']
    es_pris = calc_pris.get_eigenvalues() - results_pris['evac']

    diff = es_pris[0] - es_def[0]
    states_def = es_def + diff

    statelist = []
    [statelist.append(i) for i, state in enumerate(states_def) if (
        state < cbm and state > vbm)]

    return statelist


def return_gapstates_fix(calc_def, spin=0):
    """HOTFIX until spin-orbit works with ASR!"""

    _, calc_pris = restart('../../defects.pristine_sc/gs.gpw', txt=None)
    evac_pris = calc_pris.get_electrostatic_potential()[0,0,0]
    evac_def = calc_def.get_electrostatic_potential()[0,0,0]

    vbm, cbm = calc_pris.get_homo_lumo() - evac_pris

    es_def = calc_def.get_eigenvalues() - evac_def
    es_pris = calc_pris.get_eigenvalues() - evac_pris

    diff = es_pris[0] - es_def[0]
    states_def = es_def + diff

    statelist = []
    [statelist.append(i) for i, state in enumerate(states_def) if (
        state < cbm and state > vbm)]

    return statelist


if __name__ == '__main__':
    main.cli()
