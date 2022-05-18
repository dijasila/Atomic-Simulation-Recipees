from ase.io import read
from asr.core import command, ASRResult, prepare_result, option
import numpy as np


# def webpanel(result, row, key_description):
#
#     return None


@prepare_result
class Result(ASRResult):
    """Container for spin coherence time results."""

    coherence_function: np.ndarray

    key_descriptions = dict(
        coherence_function='Coherence function as a function of time [ms].')

    # formats = {'ase_webpanel': webpanel}


@command(module='asr.spincoherencetime',
         requires=['gs.gpw', 'structure.json'],
         dependencies=['asr.gs'],
         resources='1:1h',
         returns=Result)
@option('--pristinefile', help='Path to the pristine supercell file'
        '(needs to be of the same shape as structure.json).', type=str)
def main(pristinefile: str = 'pristine.json') -> Result:
    """Calculate spin coherence time."""
    defect = read('structure.json')
    pristine = read('pristinefile.json')
    supercell = embed_supercell(defect, pristine)
    coherence_function = get_coherence_function(supercell)

    return Result.fromdata(
        coherence_function=coherence_function)


def embed_supercell(defect, pristine):
    return defect


def get_coherence_function(structure):
    """Calculate coherence function for a given input structure."""
    return np.zeros((3, 3))


if __name__ == '__main__':
    main.cli()
