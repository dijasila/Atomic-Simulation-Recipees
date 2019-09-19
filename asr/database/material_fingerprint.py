from asr.core import command


@command(module='asr.database.material_fingerprint')
def main():
    import numpy as np
    from hashlib import md5
    import json
    from ase.io import read

    atoms = read('structure.json')
    dct = atoms.todict()
    for key, value in dct.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        dct[key] = value
    hash = md5(json.dumps(dct).encode()).hexdigest()
    results = {'asr_id': hash}
    results['__key_descriptions__'] = {'asr_id': 'KVP: Material fingerprint'}
    return results


if __name__ == '__main__':
    main.cli()
