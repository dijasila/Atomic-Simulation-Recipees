from asr.core import command


@command('asr.bader',
         dependencies=['asr.structureinfo', 'asr.gs'])
def main():
    """Calculate bader charges.

    Installation
    ------------
    To make Bader analysis we use another program. Download the executable
    for Bader analysis and put in path (this is for Linux, find the
    appropriate executable for you own OS)::

        $ mkdir baderext && cd baderext
        $ wget theory.cm.utexas.edu/henkelman/code/bader/download/
        ...bader_lnx_64.tar.gz
        $ tar -zxf bader_lnx_64.tar.gz
        $ echo  'export PATH=~/baderext:$PATH' >> ~/.bashrc
    """
    from pathlib import Path
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
    out = (Path(folder) / 'bader.out').open('w')
    err = (Path(folder) / 'bader.err').open('w')
    subprocess.run(cmd.split(), cwd=folder,
                   stdout=out,
                   stderr=err)
    out.close()
    err.close()


def print():
    """Print Bader charges."""
    import os.path as op
    fname = 'data-bader/ACF.dat'
    if not op.isfile(fname):
        return

    with open(fname) as f:
        dat = f.read()
    print(dat)


if __name__ == '__main__':
    main.cli()
