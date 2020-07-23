from asr.core import command, option, AtomsFile, DctStr
import numpy as np
from ase import Atoms


@command('asr.relax_bilayer')
@option('-a', '--atoms', help='Base layer',
        type=AtomsFile(), default='../structure.json')
@option('-s', '--settings', help='Relaxation settings',
        type=DctStr())
def main(atoms: Atoms,
         settings: dict = {'d3': True,
                           'xc': 'PBE',
                           'mode': 'interlayer'}):
    from asr.core import read_json
    from ase.io import read

    top_layer = read('toplayer.json')
    
    t_c = read_json('translation.json')['translation_vector'].astype(float)


    return
