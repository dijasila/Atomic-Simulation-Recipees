def bader():
    from ase.io import write
    from ase.units import Bohr
    from gpaw import GPAW
    from gpaw.mpi import world

    assert world.size == 1, print('Do not run in parallel!')

    gs = GPAW('gs.gpw', txt=None)
    atoms = gs.atoms
    density = gs.get_all_electron_density() * Bohr**3
    write('density.cube', atoms, data=density)

    import subprocess
    import os
    folder = 'data-bader'
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    cmd = 'bader -p all_atom -p atom_index ../density.cube'
    subprocess.run(cmd.split(), cwd=folder)


def print_results():
    with open('data-bader/ACF.dat') as f:
        dat = f.read()
    print(dat)


def get_parser():
    import argparse
    description = 'Make Bader analysis of charge density'
    parser = argparse.ArgumentParser(description=description)
    parser.set_defaults(func=main)
    return parser


def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    bader()


if __name__ == '__main__':
    main()

