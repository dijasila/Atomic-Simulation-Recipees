import typing
from asr.core import command, ASRResult, prepare_result, read_json
from pathlib import Path
import numpy as np

@prepare_result
class Result(ASRResult):
    """Descriptor(s) of bilayer."""

    descriptor: str
    full_descriptor: str

    key_descriptions = dict(descriptor='A short descriptor of a stacking',
                            full_descriptor='A full descriptor of a stacking')


def get_descriptor(folder=None):
    if folder is None:
        p = Path('.')
        folder = str(p.absolute())

    folder = [x for x in folder.split("/") if x != ""][-1]
    desc = "-".join(folder.split("-")[1:])
    return desc



@command(module='asr.bilayerdescriptor',
         returns=Result)
def main() -> Result:
    """Construct descriptors for the bilayer."""

    translation = read_json('translation.json')['translation_vector']
    transform = read_json('transformdata.json')

    rotation = transform['rotation']
    
    t_c = transform['translation'][:2] + translation

    p = "'" if not np.allclose(rotation, np.eye(3)) else "'"
    B = 'B' if not np.allclose(t_c, 0.0) else 'A'

    descriptor = 'A' + B + p

    full_descriptor = get_descriptor()


    return Result.fromdata(descriptor=descriptor,
                           full_descriptor=full_descriptor)
    

    
