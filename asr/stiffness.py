"""Stiffness tensor."""
from asr.core import command, option
from asr.result.resultdata import StiffnessResult


@command(module='asr.stiffness',
         returns=StiffnessResult)
@option('--strain-percent', help='Magnitude of applied strain.', type=float)
def main(strain_percent: float = 1.0) -> StiffnessResult:
    """Calculate stiffness tensor."""
    from asr.setup.strains import main as setupstrains
    from asr.setup.strains import get_relevant_strains, get_strained_folder_name
    from asr.relax import main as relax
    from ase.io import read
    from ase.units import J
    from asr.core import read_json, chdir
    from asr.database.material_fingerprint import main as computemf
    import numpy as np

    if not setupstrains.done:
        setupstrains(strain_percent=strain_percent)

    atoms = read('structure.json')
    ij = get_relevant_strains(atoms.pbc)

    ij_to_voigt = [[0, 5, 4],
                   [5, 1, 3],
                   [4, 3, 2]]

    links = {}
    stiffness = np.zeros((6, 6), float)
    for i, j in ij:
        dstress = np.zeros((6,), float)
        for sign in [-1, 1]:
            folder = get_strained_folder_name(sign * strain_percent, i, j)
            with chdir(folder):
                if not relax.done:
                    unrelaxed = read('unrelaxed.json')
                    relax(atoms=unrelaxed)

                if not computemf.done:
                    computemf()
            mf = read_json(folder / ('results-asr.database.'
                                     'material_fingerprint.json'))
            links[str(folder)] = mf['uid']
            structurefile = folder / 'structure.json'
            structure = read(str(structurefile))
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
    data = {}
    nd = np.sum(atoms.pbc)
    speed_of_sound_x = None
    speed_of_sound_y = None
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
        speed_of_sound_x = np.sqrt(stiffness[0, 0] / area_density)
        speed_of_sound_y = np.sqrt(stiffness[1, 1] / area_density)
    elif nd == 1:
        cell = atoms.get_cell()
        area = atoms.get_volume() / cell[2, 2]
        stiffness = stiffness[[2], :][:, [2]] * area * 1e-20
        # typical values for 1D are of the order of 10^(-10) N
    elif nd == 0:
        raise RuntimeError('Cannot compute stiffness tensor of 0D material.')

    data['speed_of_sound_x'] = speed_of_sound_x
    data['speed_of_sound_y'] = speed_of_sound_y

    for i in range(6):
        for j in range(6):
            data[f'c_{i + 1}{j + 1}'] = None
    stiffness_shape = stiffness.shape
    for i in range(stiffness_shape[0]):
        for j in range(stiffness_shape[1]):
            data[f'c_{i + 1}{j + 1}'] = stiffness[i, j]

    data['__links__'] = links
    data['stiffness_tensor'] = stiffness

    if nd == 1:
        eigs = stiffness
    else:
        eigs = np.linalg.eigvals(stiffness)
    data['eigenvalues'] = eigs
    dynamic_stability_stiffness = ['low', 'high'][int(eigs.min() > 0)]
    data['dynamic_stability_stiffness'] = dynamic_stability_stiffness
    return data


if __name__ == '__main__':
    main.cli()
