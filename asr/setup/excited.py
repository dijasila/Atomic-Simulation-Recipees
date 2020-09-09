from pathlib import Path
import numpy as np
from gpaw import GPAW
from asr.core import read_json, write_json, command, option
from gpaw.response.pair import PairDensity

# TODO:
# - restructure commands, options and default arguments
# - possible, include spin option
# - implement check for already existing folders
# - write a test


@command('asr.setup.excited')
@option('--n', help='from_state', type=str)
@option('--m', help='to_state', type=str)
# @option('--spin', help='Which spin channel is considered for excitation', type=int)
def main(n: int = 1, m: int = 1):
    """Set up folder structure and parameters for excited state calculations.

    # TODO: add thorough description of the recipe
    """
    calc = GPAW('gs.gpw', txt=None)
    create_excited_folders()

    n3 = calc.get_number_of_bands()
    Pair = PairDensity(calc)
    n1 = Pair.nocc2
    n2 = Pair.nocc1

    # spin channel 1
    occ_n_alpha = np.hstack((np.ones(n1 - n),
                             np.zeros(m),
                             np.ones(1),
                             np.zeros(n3 - (n1 - n) - m - 1)))
    occ_n_beta = np.hstack((np.ones(n2),np.zeros(n3-n2)))
    params = read_json('params.json')
    p_spin0 = params.copy()
    p_spin0['asr.gs@calculate']['calculator']['occupations'] = {'name': 'fixed', 'numbers': [occ_n_alpha, occ_n_beta]}
    p_spin0['asr.gs@calculate']['calculator']['nbands'] = n3
    p_spin0['asr.relax']['calculator']['occupations'] = {'name': 'fixed', 'numbers': [occ_n_alpha, occ_n_beta]}
    p_spin0['asr.relax']['calculator']['nbands'] = n3
    write_json('excited_spin0/params.json', p_spin0)
    # spin channel 2
    occ_n_alpha = np.hstack((np.ones(n1),np.zeros(n3-n1)))
    occ_n_beta = np.hstack((np.ones(n2-n),np.zeros(m),np.ones(1),np.zeros(n3-(n2-n)-m-1))) 
    p_spin1 = params.copy()
    p_spin1['asr.gs@calculate']['calculator']['occupations'] = {'name': 'fixed', 'numbers': [occ_n_alpha, occ_n_beta]}
    p_spin1['asr.gs@calculate']['calculator']['nbands'] = n3
    p_spin1['asr.relax']['calculator']['occupations'] = {'name': 'fixed', 'numbers': [occ_n_alpha, occ_n_beta]}
    p_spin1['asr.relax']['calculator']['nbands'] = n3
    write_json('excited_spin1/params.json', p_spin1)

    return None


def create_excited_folders():
    Path('./excited_spin0').mkdir()
    Path('./excited_spin1').mkdir()
    os.system('ln -s ./../structure.json excited_spin0/unrelaxed.json')
    os.system('ln -s ./../structure.json excited_spin1/unrelaxed.json')

    return None

if __name__ == '__main__':
    main.cli()

