# NEW excited setup recipe
from pathlib import Path
import numpy as np
# from __future__ import print_function
from ase.build import niggli_reduce
from gpaw import GPAW, PW, FermiDirac
from ase import Atoms
from ase.parallel import parprint, MPI4PY, world
import sys
import re
import math
import os
from asr.core import chdir, read_json, write_json, command, option
from ase.io import read, write
from gpaw.occupations import FixedOccupations
from gpaw import restart
from gpaw.response.pair import PairDensity

# TODO:
# - clean up imports
# - restructure commands, options and default arguments
# - possible, include spin option
# - implement check for already existing folders
# - write a test

@command('asr.setup.excited')
@option('--n', help='from_state', type=str)
@option('--m', help='to_state', type=str)
#@option('--spin', help='Which spin channel is considered for excitation', type=int)
def main(n: int = 1, m: int =  1):
    """Generate the folder structure, unrelaxed.json and paramrs.Json files for excited state.

    Generate atomic structures with displaced atoms. The generated
    atomic structures are written to 'structure.json' and put into a
    directory with the structure

        cc-{state}-{displacement%}

    """
    calc = GPAW('gs.gpw', txt=None)
    create_excited_folders()

    n3 = calc.get_number_of_bands() 
    Pair=PairDensity(calc)
    n1=Pair.nocc2
    n2=Pair.nocc1

    #spin channel 1
    occ_n_alpha = np.hstack((np.ones(n1-n),np.zeros(m),np.ones(1),np.zeros(n3-(n1-n)-m-1)))
    occ_n_beta = np.hstack((np.ones(n2),np.zeros(n3-n2)))
    params = read_json('params.json')
    p_spin0 = params.copy()
    p_spin0['asr.gs@calculate']['calculator']['occupations'] = {'name': 'fixed', 'numbers': [occ_n_alpha, occ_n_beta]}
    p_spin0['asr.gs@calculate']['calculator']['nbands'] = n3
    p_spin0['asr.relax']['calculator']['occupations'] = {'name': 'fixed', 'numbers': [occ_n_alpha, occ_n_beta]}
    p_spin0['asr.relax']['calculator']['nbands'] = n3
    write_json('excited_spin0/params.json', p_spin0)
    #spin channel 2
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

