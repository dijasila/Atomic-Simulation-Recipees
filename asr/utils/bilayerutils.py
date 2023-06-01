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


def translation(x, y, z, rotated, base, mirror):
    """Combine rotated with base by translation.
    x, y, z are cartesian coordinates.
    """
    if mirror:
       average_before = np.average(base.positions[:,2])
       base.positions[:,2] *= (-1) #here I am assuming that the z=0 plane in c2db never pass the structure because in c2db they put the structure in the middle
       average_after = np.average(base.positions[:,2])
       base.positions[:,2] += (average_before-average_after)

    stacked = base.copy()
    rotated = rotated.copy()
    rotated.translate([x, y, z])
    stacked += rotated
    stacked.wrap()

    return stacked


def construct_bilayer_inpath(blfolder, vacuum=6, interlayer=None, mag_config=None):
    from ase.io import read
    from asr.core import read_json
    import os

    # read base monolayer structure
    base = read(f'{blfolder}/../structure.json')

    # construct the top layer from monolayer
    toplayer = base.copy()
    # Increase z vacuum. Asbjorn had it and for now I am keeping it.
    # This adding in removing cell size is probably because of wraping later.
    toplayer.cell[2, 2] *= 2
    # Center atoms
    spos_av = toplayer.get_positions()
    spos_av[:, 2] += toplayer.cell[2, 2] / 4.0
    toplayer.set_positions(spos_av)

    # read the transformation data except the displacement
    U_cc = read_json(f"{blfolder}/transformdata.json")["rotation"]
    t_c = read_json(f"{blfolder}/transformdata.json")["translation"]
    # Calculate rotated and translated atoms
    spos_ac = toplayer.get_scaled_positions()
    spos_ac = np.dot(spos_ac, U_cc.T) + t_c
    # Move atoms
    toplayer.set_scaled_positions(spos_ac)
    # Wrap atoms outside of unit cell back
    toplayer.wrap(pbc=[1, 1, 1])

    toplayer.cell[2, 2] /= 2
    spos_av = toplayer.get_positions()
    spos_av[:, 2] -= toplayer.cell[2, 2] / 2
    toplayer.set_positions(spos_av)

    # Reading the interlayer distance if not provided in the input
    translation_data = read_json(f"{blfolder}/translation.json")
    if interlayer is not None:
       h = interlayer
    elif 'Saddle_from' in translation_data:
       z1 = toplayer.positions[0,2]
       saddle_atoms = read(f'{blfolder}/structure.json')
       z2 = saddle_atoms.positions[len(toplayer),2]
       h = abs(z2-z1)
       print('optimal height for saddle',h, z1, z2)
    else:
       if os.path.isfile(f"{blfolder}/results-asr.zscan.json") or os.path.isfile(f'{blfolder}/zscan_correction.txt'):
          relax_data = read_json(f"{blfolder}/results-asr.zscan.json")
          h = relax_data["optimal_height"]
       elif os.path.isfile(f"{blfolder}/results-asr.relax_bilayer.json"):
          relax_data = read_json(f"{blfolder}/results-asr.relax_bilayer.json")
          h = relax_data["optimal_height"]
   
    # Increase the vacuum of the bilayer with respect to monolayer
    # If the cell vector (vacuum) was enough for monolayer now we add w+6 for the new layer & interlayer . 
    # The average of interlayer is 3.5 so 6 is enough.
    maxz = np.max(base.positions[:, 2])
    minz = np.min(base.positions[:, 2])
    w = maxz - minz
    base.cell[2, 2] += vacuum + w
    base.cell[2, 0:2] = 0.0
    toplayer.cell = base.cell

    # Set the magnetic configuration
    try:
        magmoms = read_json(f"{blfolder}/../structure.json")[1]["magmoms"] 
    except:
        magmoms = 'NM'

    if mag_config is not None and mag_config.lower() == 'fm':
        base.set_initial_magnetic_moments(magmoms)
        toplayer.set_initial_magnetic_moments(magmoms)
    elif mag_config is not None and mag_config.lower() == 'afm':
        base.set_initial_magnetic_moments(magmoms)
        toplayer.set_initial_magnetic_moments(-magmoms)

    # Since half of the project is done I don't have mirror when it is False.
    try:
        mirror = read_json(f"{blfolder}/transformdata.json")["Bottom_layer_Mirror"] 
    except:
        mirror = False

    translation_vec = read_json(f"{blfolder}/translation.json")["translation_vector"]
    x, y = translation_vec[0], translation_vec[1]

    return translation(x, y, h, toplayer, base, mirror)


def construct_bilayer_old(path, h=None):
    from ase.io import read
    from asr.core import read_json

    top_layer = read(f'{path}/toplayer.json')
    base = read(f'{path}/../structure.json')

    t = np.array(read_json(f'{path}/translation.json')
                 ['translation_vector']).astype(float)

    if h is None:
        h = read_json(f'{path}/results-asr.relax_bilayer.json')['optimal_height']

    return translation(t[0], t[1], h, top_layer, base, mirror)


def layername(formula, nlayers, U_cc, t_c, mirror):
    s = f"{formula}-{nlayers}-{U_cc[0, 0]}_{U_cc[0, 1]}_{U_cc[1, 0]}_{U_cc[1, 1]}-"
    if np.allclose(U_cc[2, 2], -1.0):
        s += "Iz-"

    s = s + pretty_float(t_c)
    if mirror: s = 'M.' + s

    return s
