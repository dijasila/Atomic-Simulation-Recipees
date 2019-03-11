import argparse


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
    import os.path as op
    fname = 'data-bader/ACF.dat'
    if not op.isfile(fname):
        return
    
    with open(fname) as f:
        dat = f.read()
    print(dat)


def main(args=None):
    bader(**args)


short_description = 'Make Bader analysis of charge density'
parser = argparse.ArgumentParser(description=short_description)
dependencies = ['gs.py']

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)
