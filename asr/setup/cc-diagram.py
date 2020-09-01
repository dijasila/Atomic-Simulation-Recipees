"""Module for generating displaced structures for the
   configuration coordinate diagram.

The main recipe of this module is :func:`asr.setup.cc-diagram.main`

.. autofunction:: asr.setup.cc-diagram.main
"""

from pathlib import Path
from asr.core import command, option
from asr.core import write_json
import numpy as np


def create_displacements_folder(folder):
    folder.mkdir(parents=True, exist_ok=False)


@command('asr.setup.cc-diagram')
@option('--state', help='Which confguration is displaced.', type=str)
@option('--npoints', help='How many displacement points.', type=int)
def main(state: str = 'ground', npoints: int = 5):
    """Generate displaced strcutres along the interpolation bewteen
       a ground an an excited state.

    Generate atomic structures with displaced atoms. The generated
    atomic structures are written to 'structure.json' and put into a
    directory with the structure

        cc-{state}-{displacement%}

    """
    from ase.parallel import world
    from gpaw import restart

    name_1 = 'gs.gpw'
    name_2 = 'ex.gpw'

    if state == 'excited':
        name_1 = 'ex.gpw'
        name_2 = 'gs.gpw'

    atoms_1, calc = restart(name_1, txt=None)
    atoms_2, _ = restart(name_2, txt=None)

    folders = []

    delta_r = atoms_2.positions - atoms_1.positions

    displ_n = np.linspace(-1.0, 1.0, npoints, endpoint=True)
    m_a = atoms_1.get_masses()
    pos_ai = atoms_1.positions.copy()

    for displ in displ_n:
        Q = (((displ * delta_r)**2).sum(axis=-1) * m_a).sum()

        folder = Path('cc-' + state + '-{}%'.format(int(displ * 100)))

        if world.rank == 0:
            create_displacements_folder(folder)

        atoms_1.positions += displ * delta_r
        atoms_1.write(folder / 'structure.json')
        folders.append(str(folder))

        atoms_1.positions = pos_ai

        params = {'Q': Q, 'displ': displ}
        write_json(folder / 'params.json', params)

    world.barrier()
    return {'folders': folders}
