from io import StringIO

import spglib
from ase import Atoms


def get_symmetry_id(atoms, symprec):
    import bulk_enumerator as be
    cell, spos, numbers = spglib.standardize_cell(atoms,
                                                  symprec=symprec)
    standard_atoms = Atoms(numbers=numbers, cell=cell, scaled_positions=spos,
                           pbc=atoms.pbc)
    poscar_buffer = StringIO()
    standard_atoms.write(poscar_buffer, format='vasp')
    b = be.bulk.BULK()
    b.set_structure_from_file(poscar_buffer.getvalue())
    poscar_buffer.close()
    name = b.get_name()
    return name
