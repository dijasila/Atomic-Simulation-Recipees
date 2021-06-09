import numpy as np


def pretty_float(arr):
    f1 = round(arr[0], 2)
    if np.allclose(f1, 0.0):
        s1 = "0"
    else:
        s1 = str(f1)
    f2 = round(arr[1], 2)
    if np.allclose(f2, 0.0):
        s2 = "0"
    else:
        s2 = str(f2)

    return f'{s1}_{s2}'


def translation(x, y, z, rotated, base):
    """Combine rotated with base by translation.

    x, y, z are cartesian coordinates.
    """
    stacked = base.copy()
    rotated = rotated.copy()
    rotated.translate([x, y, z])
    stacked += rotated
    stacked.wrap()

    return stacked


def construct_bilayer(path, h=None):
    from ase.io import read
    from asr.core import read_json

    top_layer = read(f'{path}/toplayer.json')
    base = read(f'{path}/../structure.json')

    t = np.array(read_json(f'{path}/translation.json')
                 ['translation_vector']).astype(float)

    if h is None:
        h = read_json(f'{path}/results-asr.relax_bilayer.json')['optimal_height']

    return translation(t[0], t[1], h, top_layer, base)


def layername(formula, nlayers, U_cc, t_c):
    s = f"{formula}-{nlayers}-{U_cc[0, 0]}_{U_cc[0, 1]}_{U_cc[1, 0]}_{U_cc[1, 1]}-"
    if np.allclose(U_cc[2, 2], -1.0):
        s += "Iz-"

    s = s + pretty_float(t_c)

    return s
