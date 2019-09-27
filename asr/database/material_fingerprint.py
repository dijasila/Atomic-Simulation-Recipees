from asr.core import command


def todict(atoms):
    import numpy as np
    d = dict(atoms.arrays)
    d['cell'] = np.asarray(atoms.cell)
    d['pbc'] = atoms.pbc
    if atoms._celldisp.any():
        d['celldisp'] = atoms._celldisp
    if atoms.constraints:
        d['constraints'] = atoms.constraints
    if atoms.info:
        d['info'] = atoms.info
    return d


@command(module='asr.database.material_fingerprint')
def main():
    import numpy as np
    from hashlib import md5
    import json
    from ase.io import read
    from collections import OrderedDict

    atoms = read('structure.json')
    dct = todict(atoms)

    for key, value in dct.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dct[key] = value

    # Make sure that that keys appear in order
    orddct = OrderedDict()
    keys = list(dct.keys())
    keys.sort()
    for key in keys:
        orddct[key] = dct[key]

    hash = md5(json.dumps(orddct).encode()).hexdigest()
    formula = atoms.symbols.formula
    results = {'asr_id': hash,
               'uid': f'{formula:abc}-' + hash[:12]}
    results['__key_descriptions__'] = {'asr_id': 'KVP: Material fingerprint',
                                       'uid': 'KVP: Unique identifier'}
    return results


if __name__ == '__main__':
    main.cli()
