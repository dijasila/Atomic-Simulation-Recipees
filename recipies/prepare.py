def prepare(filename, destination, vacuum=7.5):
    from ase.io import read, write
    from ase.build import niggli_reduce
    import numpy as np
    atoms = read(filename)
    pbc = atoms.get_pbc()
    if not np.all(pbc):
        axis = np.argwhere(~pbc)[0]
        atoms.center(vacuum=vacuum,
                     axis=axis)
    atoms.pbc = (1, 1, 1)
    niggli_reduce(atoms)
    atoms.pbc = pbc
    write(destination, atoms)
    
    
def get_parser():
    import argparse
    description = 'Prepare structure for recipies'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--filename', default='origin.traj',
                        help='Path to original structure')
    parser.add_argument('--destination', default='start.traj')
    parser.add_argument('--vacuum', default=7.5, type=float,
                        help='Vacuum to add to nonperiodic directions')
    return parser


def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    prepare(**args)

        
if __name__ == '__main__':
    main()
