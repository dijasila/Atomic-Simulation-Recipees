"""Bader charge analysis."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.io import write
from ase.units import Bohr

from asr.core import ASRResult, command, prepare_result
from asr.database.browser import (describe_entry, entry_parameter_description,
                                  href, make_panel_description)

panel_description = make_panel_description(
    """The Bader charge analysis ascribes a net charge to an atom
by partitioning the electron density according to its zero-flux surfaces.""",
    articles=[
        href("""W. Tang et al. A grid-based Bader analysis algorithm
without lattice bias. J. Phys.: Condens. Matter 21, 084204 (2009).""",
             'https://doi.org/10.1088/0953-8984/21/8/084204')])


def webpanel(result, row, key_descriptions):
    rows = [[str(a), symbol, f'{charge:.2f}']
            for a, (symbol, charge)
            in enumerate(zip(result.sym_a, result.bader_charges))]
    table = {'type': 'table',
             'header': ['Atom index', 'Atom type', 'Charge (|e|)'],
             'rows': rows}

    parameter_description = entry_parameter_description(
        row.data,
        'asr.bader')

    title_description = panel_description + parameter_description

    panel = {'title': describe_entry('Bader charges',
                                     description=title_description),
             'columns': [[table]]}

    return [panel]


@prepare_result
class Result(ASRResult):

    bader_charges: np.ndarray
    sym_a: List[str]

    key_descriptions = {'bader_charges': 'Array of charges [\\|e\\|].',
                        'sym_a': 'Chemical symbols.'}

    formats = {"ase_webpanel": webpanel}


@command('asr.bader',
         dependencies=['asr.gs'],
         returns=Result)
def main() -> Result:
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
    return Result(data=dict(bader_charges=charges,
                            sym_a=sym_a))


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
