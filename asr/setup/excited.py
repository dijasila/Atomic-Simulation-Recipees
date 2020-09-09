from pathlib import Path
import numpy as np
import os
from gpaw import GPAW
from asr.core import read_json, write_json, command, option
from gpaw.response.pair import PairDensity

# TODO:
# - restructure commands, options and default arguments
# - possibly, include spin option
# - write a test


@command('asr.setup.excited')
@option('--n', help='State from which an electron will be removed. 1'
        ' corresponds to the highest occupied state, 2 corresponds to the'
        ' second highest occupied state, and so on.', type=int)
@option('--m', help='Unoccupied state into which the removed atom will be'
        ' placed. 1 corresponds to the lowest unoccupied state, 2'
        'corresponds to the second lowest unoccupied stated, and so on.',
        type=int)
@option('--spin', help='Specify which spin channel you want to excite the'
        ' system in. Choose 0 for the first spin channel, 1 for the second'
        ' spin channel, and 2 if both should be considered.')
def main(n: int = 1, m: int = 1, spin: int = 2):
    """Set up folder structure and parameters for excited state calculations.

    This recipe creates two folders for calculations of excited states. Run
    the recipe inside a folder where a finished groundstate calculation,
    relaxed structure of the groundstate, and params.json is present. Within
    the newly created folders, a unrelaxed.json structure gets linked (which
    is the relaxed ground state structure), as well as writing a params.json
    file containing all of the parameters from the parent groundstate calcu-
    lation plus parameters for fixed occupations and number of bands that are
    needed for excited state calculations.
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
    occ_n_beta = np.hstack((np.ones(n2), np.zeros(n3 - n2)))
    params = read_json('params.json')
    p_spin0 = params.copy()
    p_spin0['asr.gs@calculate']['calculator']['occupations'] = {'name': 'fixed',
                                                                'numbers':
                                                                [occ_n_alpha,
                                                                    occ_n_beta]}
    p_spin0['asr.gs@calculate']['calculator']['nbands'] = n3
    p_spin0['asr.relax']['calculator']['occupations'] = {'name': 'fixed',
                                                         'numbers':
                                                         [occ_n_alpha, occ_n_beta]}
    p_spin0['asr.relax']['calculator']['nbands'] = n3
    write_json('excited_spin0/params.json', p_spin0)
    # spin channel 2
    occ_n_alpha = np.hstack((np.ones(n1), np.zeros(n3 - n1)))
    occ_n_beta = np.hstack((np.ones(n2 - n), np.zeros(m), np.ones(1),
                            np.zeros(n3 - (n2 - n) - m - 1)))
    p_spin1 = params.copy()
    p_spin1['asr.gs@calculate']['calculator']['occupations'] = {'name': 'fixed',
                                                                'numbers':
                                                                [occ_n_alpha,
                                                                    occ_n_beta]}
    p_spin1['asr.gs@calculate']['calculator']['nbands'] = n3
    p_spin1['asr.relax']['calculator']['occupations'] = {'name': 'fixed',
                                                         'numbers':
                                                         [occ_n_alpha, occ_n_beta]}
    p_spin1['asr.relax']['calculator']['nbands'] = n3
    write_json('excited_spin1/params.json', p_spin1)

    return None


def create_excited_folders():
    if not (Path('./excited_spin0').is_dir()
            and Path('./excited_spin1').is_dir()):
        Path('./excited_spin0').mkdir()
        Path('./excited_spin1').mkdir()
        os.system('ln -s ./../structure.json excited_spin0/unrelaxed.json')
        os.system('ln -s ./../structure.json excited_spin1/unrelaxed.json')
    else:
        print('WARNING: excited folders already exist! Overwrite params.json'
              ' and keep already linked strucrures.')

    return None


if __name__ == '__main__':
    main.cli()
