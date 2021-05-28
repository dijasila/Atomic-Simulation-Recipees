from asr.core import command, option, AtomsFile, ASRResult
from ase import Atoms
import numpy as np
from asr.utils.bilayerutils import layername


def stack(stacked, base, U_cc, t_c, tvec, h):
    from asr.utils.bilayerutils import translation as stack_trans
    new_layer = base.copy()
    spos_ac = new_layer.get_scaled_positions()
    spos_ac = np.dot(spos_ac, U_cc.T) + t_c
    new_layer.set_scaled_positions(spos_ac)
    new_layer.wrap(pbc=[1, 1, 1])
    return stack_trans(tvec[0], tvec[1], h, new_layer, stacked), new_layer


def inflate_vacuum(atoms, height, nlayers):
    """Inflate vacuum such that there is room for nlayers.

    Inflate vacuum enough so that nlayers can fit into the
    unit cell.

    The vacuum, defined as u_v[2] / 2 (half the z-component
    of the third lattice vector), should contain the nlayers
    and additional vacuum equal to the original amount of vacuum.

    Therefore the new vacuum should be

    2 * new_vacuum = 2 * old_vacuum + (nlayers-1) * height
    or
    uz -> uz + (nlayers-1)*height
    """
    atoms = atoms.copy()

    uz = atoms.cell[2, 2]

    uz += (nlayers - 1) * height

    atoms.cell[2, 2] = uz

    return atoms


@command('asr.stack_multilayer')
@option('-a', '--atoms', help='Base layer',
        type=AtomsFile(), default='../structure.json')
@option('-l', '--height', help='Interlayer stacking height',
        type=float, default=None)
@option('-n', '--nlayers', help='Number of layers',
        type=int, default=2)
def main(atoms: Atoms,
         height: float,
         nlayers: int) -> ASRResult:
    from asr.core import read_json
    import os
    transform_data = read_json('transformdata.json')
    translation = np.array(read_json('translation.json')['translation_vector'])

    U_cc = transform_data['rotation']
    t_c = transform_data['translation']

    height = height or read_json('results-asr.relax_bilayer.json')['optimal_height']

    stacked = inflate_vacuum(atoms, height, nlayers)

    for n in range(1, nlayers):
        stacked, rotated = stack(stacked, atoms,
                                 U_cc, t_c, translation * n,
                                 height * n)
        atoms = rotated

    f = atoms.get_chemical_formula()
    t = atoms.cell.scaled_positions(
        np.array([translation[0], translation[1], 0.0])) + t_c
    foldername = layername(f, nlayers, U_cc, t)
    if not os.path.isdir(foldername):
        os.mkdir(foldername)

    stacked.write(f'{foldername}/structure.json')


if __name__ == '__main__':
    main.cli()
