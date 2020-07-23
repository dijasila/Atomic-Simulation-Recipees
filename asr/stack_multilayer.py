from asr.core import command, option, AtomsFile
from ase import Atoms
import numpy as np


def stack(atoms, base, U_cc, t_c, tvec, h):
    from asr.stack_bilayer import translation
    new_layer = base.copy()
    spos_ac = new_layer.get_scaled_positions()
    spos_ac = np.dot(spos_ac, U_cc.T) + t_c
    new_layer.set_scaled_positions(spos_ac)
    new_layer.wrap(pbc=[1, 1, 1])

    return translation(tvec[0], tvec[1], h, new_layer, atoms)


@command('asr.stack_multilayer')
@option('-a', '--atoms', help='Base layer',
        type=AtomsFile(), default='../structure.json')
@option('-h', '--height', help='Stacking height',
        type=float, default=None)
def main(atoms: Atoms,
         height):
    from asr.core import read_json
    transform_data = read_json('transformdata.json')
    translation = read_json('translation.json')['translation_vector'].astype(float)

    
    U_cc = transform_data['rotation']
    t_c = transform_data['translation']

    
    height = height or read_json('results-asr.relax_bilayer.json')['optimal_height']

