from asr.core import command, option

tests = [{'cli': ['ase build -x diamond Si structure.json',
                  'asr run "setup.strains --kptdensity 2.0"',
                  'asr run "setup.params asr.relax:ecut 200" strains*/',
                  'asr run "relax --nod3" strains*/',
                  'asr run stiffness']}]


@command(module='asr.stiffness',
         tests=tests)
@option('--strain-percent', help='Magnitude of applied strain')
def main(strain_percent=1.0):
    from asr.setup.strains import (get_strained_folder_name,
                                   get_relevant_strains)
    from ase.io import read
    from ase.units import J
    import numpy as np
    
    atoms = read('structure.json')
    ij = get_relevant_strains(atoms.pbc)

    ij_to_voigt = [[0, 5, 4],
                   [5, 1, 3],
                   [4, 3, 2]]

    stiffness = np.zeros((6, 6), float)
    for i, j in ij:
        dstress = np.zeros((6,), float)
        for sign in [-1, 1]:
            folder = get_strained_folder_name(sign * strain_percent, i, j)
            structure = read(str(folder / 'structure.json'))
            # The structure already has the stress if it was
            # calculated
            stress = structure.get_stress(voigt=True)
            dstress += stress * sign
        stiffness[:, ij_to_voigt[i][j]] = dstress / (strain_percent * 0.02)

    stiffness = np.array(stiffness, float)
    # We work with Mandel notation which is conventional and convenient
    stiffness[3:, :] *= 2**0.5
    stiffness[:, 3:] *= 2**0.5

    # Convert the stiffness tensor from [eV/Ang^3] -> [J/m^3]=[N/m^2]
    stiffness *= 10**30 / J

    # Now do some post processing
    data = {'__key_descriptions__': {}}
    kd = data['__key_descriptions__']
    nd = np.sum(atoms.pbc)
    if nd == 2:
        cell = atoms.get_cell()
        # We have to normalize with the supercell size
        z = cell[2, 2]
        stiffness = stiffness[[0, 1, 5], :][:, [0, 1, 5]] * z * 1e-10
        from ase.units import kg
        from ase.units import m as meter
        area = atoms.get_volume() / cell[2, 2]
        mass = sum(atoms.get_masses())
        area_density = (mass / kg) / (area / meter**2)
        # speed of sound in m/s
        speed_x = np.sqrt(stiffness[0, 0] / area_density)
        speed_y = np.sqrt(stiffness[1, 1] / area_density)
        speed_of_sound = np.array([speed_x, speed_y])
        data['speed_of_sound'] = speed_of_sound
        kd['speed_of_sound'] = 'KVP: Speed of sound [m/s]'
        kd['stiffness_tensor'] = 'Stiffness tensor [N/m]'
    elif nd == 1:
        area = atoms.get_volume() / cell[2, 2]
        stiffness = stiffness[5, 5] * area * 1e-20
        kd['stiffness_tensor'] = 'Stiffness tensor [N]'
    else:
        kd['stiffness_tensor'] = 'Stiffness tensor [N/m^2]'

    data['stiffness_tensor'] = stiffness.tolist()

    return data
