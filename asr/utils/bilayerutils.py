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
    stacked = base.copy()
    rotated = rotated.copy()
    rotated.translate([x, y, z])
    stacked += rotated
    stacked.wrap()

    return stacked


def layername(formula, nlayers, U_cc, t_c): 
    s = f"{formula}-{nlayers}-{U_cc[0, 0]}_{U_cc[0, 1]}_{U_cc[1, 0]}_{U_cc[1, 1]}-"
    if np.allclose(U_cc[2, 2], -1.0):
        s = s + "Iz-"
        
    s = s + pretty_float(t_c)

    return s
