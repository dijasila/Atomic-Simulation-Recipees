from io import StringIO

import numpy as np
import spglib
from ase import Atoms


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
    print(dir(b))
    name = b.get_name()
    return name
