from pathlib import Path
import numpy as np
import os
from gpaw import restart
from ase.io import Trajectory
from asr.core import write_json, command, option


@command('asr.setup.excited')
@option('--n', help='State from which an electron will be removed. 1'
        ' corresponds to the highest occupied state, 2 corresponds to the'
        ' second highest occupied state, and so on.', type=int)
@option('--m', help='Unoccupied state into which the removed atom will be'
        ' placed. 1 corresponds to the lowest unoccupied state, 2'
        'corresponds to the second lowest unoccupied stated, and so on.',
        type=int)
@option('--setup-smart/--setup-any', help='Flag to only set up excitations '
        'that are above a particular energy threshold (0.5 eV). If '
        '--setup-any is selected, any (n -> m) excitation will be created.',
        is_flag=True)
def main(n: int = 1,
         m: int = 1,
         setup_smart: bool = True):
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
    if n == 1 and m == 1:
        print('INFO: excite HOMO -> LUMO transition.')
    else:
        print('INFO: excite electron from {}. highest occupied to the {}. '
              'lowest unoccupied orbital.'.format(n, m))

    atoms, calc = restart('gs.gpw', txt=None)

    N_tot = calc.get_number_of_bands()
    E_F = calc.get_fermi_level()

    # set up lists of occupied eigenvalues for either one or both spins
    set_both = False
    set_excitation = True
    occ_spin0 = []
    ev_spin0 = calc.get_eigenvalues(spin=0)
    [occ_spin0.append(en) for en in ev_spin0 if en < E_F]
    n1 = len(occ_spin0)
    # check for threshold in excitation for first spin channel
    if (ev_spin1[n1+1] - ev_spin1[n1]) < 0.5 and setup_smart:
        set_excitation = False
    if calc.get_number_of_spins() == 2:
        occ_spin1 = []
        ev_spin1 = calc.get_eigenvalues(spin=1)
        [occ_spin1.append(en) for en in ev_spin1 if en < E_F]
        n2 = len(occ_spin1)
        # check for threshold in excitation for second spin channel
        if (ev_spin1[n2+1] - ev_spin1[n2]) < 0.5 and setup_smart:
            set_excitation = False
        # if occupations for both spin channels are the same, only set up the
        # first one (even though we have a spin-polarized calculation
        if n1 != n2:
            set_both = True
            print('INFO: set up occupations for both spin channels.')
        else:
            print('INFO: symmetric occupations in both spin channels. '
                  'Only set up one folder for the first channel.')

    if set_excitation:
        # extract old calculator parameters
        params_relax = Trajectory('relax.traj')[-1].get_calculator().todict()
        params_gs = calc.todict()
        params_relax['name'] = 'gpaw'
        params_gs['name'] = 'gpaw'
        newparams = {'asr.gs@calculate': {'calculator': params_gs},
                     'asr.relax': {'calculator': params_relax}}

        # create occupations array for the first spin channel
        occ_n_alpha = np.hstack((np.ones(n1 - n),
                                 np.zeros(1),
                                 np.ones(n - 1),
                                 np.zeros(m - 1),
                                 np.ones(1),
                                 np.zeros(N_tot - n1 - m)))
        occ_n_beta = np.hstack((np.ones(n1), np.zeros(N_tot - n1)))
        p_spin0 = newparams.copy()
        p_spin0['asr.gs@calculate']['calculator']['occupations'] = {'name': 'fixed',
                                                                    'numbers':
                                                                    [occ_n_alpha,
                                                                        occ_n_beta]}
        p_spin0['asr.gs@calculate']['calculator']['nbands'] = N_tot
        p_spin0['asr.relax']['calculator']['occupations'] = {'name': 'fixed',
                                                             'numbers':
                                                             [occ_n_alpha, occ_n_beta]}
        p_spin0['asr.relax']['calculator']['nbands'] = N_tot
        create_excited_folders(0)
        write_json('excited_spin0/params.json', p_spin0)
        # for spin-polarized calculations, also create occupations for the second
        # spin channel
        if calc.get_number_of_spins() == 2 and set_both:
            occ_n_alpha = np.hstack((np.ones(n2), np.zeros(N_tot - n2)))
            occ_n_beta = np.hstack((np.ones(n2 - n),
                                    np.zeros(1),
                                    np.ones(n - 1),
                                    np.zeros(m - 1),
                                    np.ones(1),
                                    np.zeros(N_tot - n2 - m)))
            p_spin1 = newparams.copy()
            p_spin1['asr.gs@calculate']['calculator']['occupations'] = {'name': 'fixed',
                                                                        'numbers':
                                                                        [occ_n_alpha,
                                                                            occ_n_beta]}
            p_spin1['asr.gs@calculate']['calculator']['nbands'] = N_tot
            p_spin1['asr.relax']['calculator']['occupations'] = {'name': 'fixed',
                                                                 'numbers':
                                                                 [occ_n_alpha, occ_n_beta]}
            p_spin1['asr.relax']['calculator']['nbands'] = N_tot
            create_excited_folders(1)
            write_json('excited_spin1/params.json', p_spin1)
    elif not set_excitation:
        print("WARNING: chosen excitation is below the physically reasonable threshold. "
              "Don't create the excitation! If wanted, try setting up another excitation.")

    return None


def create_excited_folders(channel):
    foldername = './excited_spin{}'.format(int(channel))
    if not Path(foldername).is_dir():
        Path(foldername).mkdir()
        os.system('ln -s ./../structure.json '
                  'excited_spin{}/unrelaxed.json'.format(int(channel)))
    else:
        print('WARNING: excited folder already exists! Overwrite params.json'
              ' in {}/ and keep already linked structures.'.format(
                  foldername))

    return None


if __name__ == '__main__':
    main.cli()
