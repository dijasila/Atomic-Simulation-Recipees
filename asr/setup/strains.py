from asr.core import command, option

tests = [{'cli': ['ase build -x diamond Si.json structure.json',
                  'asr run setup.strains']}]


@command('asr.setup.strains')
@option('--strain-percent', help='Strain percentage')
def main(strain_percent=1):
    from ase.io import read
    import numpy as np
    from pathlib import Path

    atoms = read('structure.json')
    cell_cv = atoms.get_cell()
    pbc = atoms.pbc
    if np.sum(pbc) == 3:
        ij = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    elif np.sum(pbc) == 2:
        ij = ((0, 0), (1, 1), (0, 1))
    elif np.sum(pbc) == 1:
        ij = ((2, 2))

    itov_i = ['x', 'y', 'z']
    for i, j in ij:
        strain_vv = np.eye(3)
        strain_vv[i, j] = strain_percent / 100.0
        strain_vv = (strain_vv + strain_vv.T) / 2
        strained_cell_cv = np.dot(cell_cv, strain_vv)
        atoms.set_cell(strained_cell_cv, scale_atoms=True)
        name = itov_i[i] + itov_i[j]
        folder = Path(f'strains-{name}-{strain_percent}/')
        folder.mkdir()
        atoms.write(str(folder / 'structure.json'))
