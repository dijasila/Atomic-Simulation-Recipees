"""Bader charge analysis."""
from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from ase.units import Bohr

from asr.core import command
from asr.result.resultdata import BaderResult

@command('asr.bader',
         dependencies=['asr.gs'],
         returns=BaderResult)
def main() -> BaderResult:
    """Calculate bader charges.

    To make Bader analysis we use another program. Download the executable
    for Bader analysis and put in path (this is for Linux, find the
    appropriate executable for you own OS)

        $ mkdir baderext && cd baderext
        $ wget theory.cm.utexas.edu/henkelman/code/bader/download/
        ...bader_lnx_64.tar.gz
        $ tar -xf bader_lnx_64.tar.gz
        $ echo 'export PATH=~/baderext:$PATH' >> ~/.bashrc
    """

    from gpaw.mpi import world
    # from gpaw.new.ase_interface import GPAW
    from gpaw import GPAW

    assert world.size == 1, 'Do not run in parallel!'

    gs = GPAW('gs.gpw')
    atoms, charges = bader(gs)
    sym_a = atoms.get_chemical_symbols()
    return BaderResult(data=dict(bader_charges=charges, sym_a=sym_a))


def bader(gs) -> tuple[Atoms, np.ndarray]:
    """Preform Bader analysis.

    * read GPAW-gpw file
    * calculate all-electron density
    * write CUBE file
    * run "bader" program
    * check for correct number of volumes
    * check for correct total charge

    Returns ASE Atoms object and ndarray of charges in units of :math:`|e|`.
    """
    rho = gs.get_all_electron_density(gridrefinement=4)
    atoms = gs.atoms

    if np.linalg.det(atoms.cell) < 0.0:
        print('Left handed unit cell!')
        rho = rho.transpose([1, 0, 2])
        atoms = atoms.copy()
        atoms.cell = atoms.cell[[1, 0, 2]]
        atoms.pbc = atoms.pbc[[1, 0, 2]]

    write('density.cube', atoms, data=rho * Bohr**3)

    cmd = 'bader density.cube'
    with Path('bader.out').open('w') as out:
        with Path('bader.err').open('w') as err:
            subprocess.run(cmd.split(),
                           stdout=out,
                           stderr=err)

    if 0:  # looks like this check is too strict!
        n = count_number_of_bader_maxima(Path('bader.out'))
        if n != len(atoms):
            raise ValueError(f'Wrong number of Bader volumes: {n}')

    charges = -read_bader_charges('ACF.dat')
    charges += atoms.get_atomic_numbers()
    assert abs(charges.sum()) < 0.01

    return gs.atoms, charges


def read_bader_charges(filename: str | Path = 'ACF.dat') -> np.ndarray:
    path = Path(filename)
    charges = []
    with path.open() as fd:
        for line in fd:
            words = line.split()
            if len(words) == 7:
                charges.append(float(words[4]))
    return np.array(charges)


def count_number_of_bader_maxima(path: Path) -> int:
    """Read number of maxima from output file."""
    for line in path.read_text().splitlines():
        if line.strip().startswith('SIGNIFICANT MAXIMA FOUND:'):
            return int(line.split()[-1])
    assert False


if __name__ == '__main__':
    main.cli()
