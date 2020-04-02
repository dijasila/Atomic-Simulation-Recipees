"""Module for generating atomic structures with displaced atoms.

The main recipe of this module is :func:`asr.setup.displacements.main`

.. autofunction:: asr.setup.displacements.main
"""

from pathlib import Path
from asr.core import command, option


def get_displacement_folder(atomic_index,
                            cartesian_index,
                            displacement_sign,
                            displacement):
    """Generate folder name from (ia, iv, sign, displacement)."""
    cartesian_symbol = 'xyz'[cartesian_index]
    displacement_symbol = ' +-'[displacement_sign]
    foldername = (f'{displacement}-{atomic_index}'
                  f'-{displacement_symbol}{cartesian_symbol}')
    folder = Path('displacements') / foldername
    return folder


def create_displacements_folder(folder):
    folder.mkdir(parents=True, exist_ok=False)


def get_all_displacements(atoms):
    """Generate ia, iv, sign for all displacements."""
    for ia in range(len(atoms)):
        for iv in range(3):
            for sign in [-1, 1]:
                yield (ia, iv, sign)


def displace_atom(atoms, ia, iv, sign, delta):
    new_atoms = atoms.copy()
    pos_av = new_atoms.get_positions()
    pos_av[ia, iv] += sign * delta
    new_atoms.set_positions(pos_av)
    return new_atoms


@command('asr.setup.displacements')
@option('--displacement', help='How much to displace atoms.')
@option('--copy-params', help='Copy params.json to displacement folders.')
def main(displacement=0.01, copy_params=True):
    """Generate atomic displacements.

    Generate atomic structures with displaced atoms. The generated
    atomic structures are written to 'structure.json' and put into a
    directory with the structure

        displacements/{displacement}-{atomic_index}-{displacement_symbol}{cartesian_symbol}

    Notice that all generated directories are a sub-directory of displacements/.

    """
    from ase.io import read
    from ase.parallel import world
    structure = read('structure.json')
    folders = []
    params = Path('params.json')
    if not params.is_file():
        copy_params = False
    else:
        params_text = params.read_text()

    for ia, iv, sign in get_all_displacements(structure):
        folder = get_displacement_folder(ia, iv,
                                         sign, displacement)
        create_displacements_folder(folder)
        new_structure = displace_atom(structure, ia, iv, sign, displacement)
        new_structure.write(folder / 'structure.json')
        folders.append(str(folder))

        if copy_params and params.is_file() and world.rank == 0:
            (folder / 'params.json').write_text(params_text)

    return {'folders': folders}
