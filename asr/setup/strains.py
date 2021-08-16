"""Generate strained atomic structures."""
from ase import Atoms
# from asr.core import command, option, atomsopt


def get_relevant_strains(pbc):
    import numpy as np
    if np.sum(pbc) == 3:
        ij = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    elif np.sum(pbc) == 2:
        ij = ((0, 0), (1, 1), (0, 1))
    elif np.sum(pbc) == 1:
        ij = ((2, 2), )
    return ij


# @command('asr.setup.strains')
# @atomsopt
# @option('--strain-percent', help='Strain percentage', type=float)
# @option('-i', '--i', type=int, help='Strain component=i of cell.')
# @option('-j', '--j', type=int, help='Strain component=j of cell.')
def main(
        atoms: Atoms,
        strain_percent: float = 1,
        i: int = 0,
        j: int = 0,
) -> Atoms:
    import numpy as np

    atoms = atoms.copy()
    cell_cv = atoms.get_cell()

    strain_vv = np.eye(3)
    strain_vv[i, j] += strain_percent / 100.0
    strain_vv = (strain_vv + strain_vv.T) / 2
    strained_cell_cv = np.dot(cell_cv, strain_vv)
    atoms.set_cell(strained_cell_cv, scale_atoms=True)

    return atoms


if __name__ == '__main__':
    main.cli()
