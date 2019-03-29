import json
from io import StringIO
from pathlib import Path

import click
import numpy as np
import spglib

from ase import Atoms
from ase.io import read


def get_symmetry_id(atoms, symprec):
    import bulk_enumerator as be
    cell, spos, numbers = spglib.standardize_cell(atoms,
                                                  symprec=symprec)
    print(cell.dot(np.linalg.inverse(cell)))
    standard_atoms = Atoms(numbers=numbers, cell=cell, scaled_positions=spos,
                           pbc=atoms.pbc)
    poscar_buffer = StringIO()
    standard_atoms.write(poscar_buffer, format='vasp')
    b = be.bulk.BULK()
    b.set_structure_from_file(poscar_buffer.getvalue())
    poscar_buffer.close()
    name = b.get_name()
    return name


@click.command()
@click.option('-a', '--atomfile', type=str,
              help='Atomic structure',
              default='start.json')
@click.option('-s', '--symprec', type=float,
              help='Symmetry precision.',
              default=0.5)
def main(atomfile, symprec):
    atoms = read(atomfile)
    name = get_symmetry_id(atoms, symprec)
    Path('prototype.json').write_text(
        json.dumps({'prototype': name}))


def collect_data():
    kvp = json.loads(Path('prototype.json'))
    return kvp
