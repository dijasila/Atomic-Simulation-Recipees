from asr.core import command, argument, option
import numpy as np


def normalize_nonpbc_atoms(atoms1, atoms2):
    atoms1, atoms2 = atoms1.copy(), atoms2.copy()

    pbc1_c = atoms1.get_pbc()
    pbc2_c = atoms2.get_pbc()

    assert all(pbc1_c == pbc2_c)

    cell2_cv = atoms2.get_cell()
    cell2_cv[~pbc2_c] = atoms1.get_cell()[~pbc1_c]
    atoms2.set_cell(cell2_cv)
    return atoms1, atoms2


def get_rmsd(atoms1, atoms2, adaptor=None, matcher=None):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from ase.build import niggli_reduce

    if adaptor is None:
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()

    if matcher is None:
        matcher = StructureMatcher(primitive_cell=False, attempt_supercell=True)

    atoms1, atoms2 = normalize_nonpbc_atoms(atoms1, atoms2)

    pbc_c = atoms1.get_pbc()
    atoms1 = atoms1.copy()
    atoms2 = atoms2.copy()
    atoms1.set_pbc(True)
    atoms2.set_pbc(True)
    niggli_reduce(atoms1)
    niggli_reduce(atoms2)
    struct1 = adaptor.get_structure(atoms1)
    struct2 = adaptor.get_structure(atoms2)

    struct1, struct2 = matcher._process_species([struct1, struct2])
    if not matcher._subset and matcher._comparator.get_hash(struct1.composition) \
            != matcher._comparator.get_hash(struct2.composition):
        return None

    struct1, struct2, fu, s1_supercell = matcher._preprocess(struct1, struct2)
    match = matcher._match(struct1, struct2, fu, s1_supercell,
                           break_on_match=True)
    if match is None:
        return None
    else:
        rmsd = match[0]
        # Fix normalization
        vol = atoms1.get_volume()
        natoms = len(atoms1)
        old_norm = (natoms / vol)**(1 / 3)
        rmsd /= old_norm  # Undo
        lenareavol = np.linalg.det(atoms1.get_cell()[pbc_c][:, pbc_c])
        new_norm = (natoms / lenareavol)**(1 / sum(pbc_c))
        rmsd *= new_norm  # Apply our own norm
        return rmsd


@command(module='asr.database.rmsd',
         resources='1:20m',
         save_results_file=False)
@argument('database')
@option('-r', '--rmsd-tol', help='RMSD tolerance.')
def main(database, rmsd_tol=1):
    """Take an input database filter out duplicates.

    Uses asr.duplicates.check_duplicates.

    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.db import connect
    db = connect(database)
    adaptor = AseAtomsAdaptor()
    matcher = StructureMatcher(primitive_cell=False,
                               attempt_supercell=True,
                               stol=rmsd_tol)

    rows = {}
    for row in db.select(include_data=False):
        rows[row.id] = {'atoms': row.toatoms(),
                        'row': row}

    rmsd_by_id = {}
    for rowid, row in rows.items():
        atoms = row.toatoms()
        row_rmsd_by_id = {}
        for otherrowid, otherrow in rows.items():
            if rowid == otherrowid:
                continue
            if not row.formula == otherrow.formula:
                continue
            rmsd = get_rmsd(atoms, otherrow.toatoms(),
                            adaptor=adaptor,
                            matcher=matcher)
            if rmsd is None:
                continue
            row_rmsd_by_id[otherrowid] = rmsd
        rmsd_by_id[rowid] = row_rmsd_by_id
    return rmsd_by_id


if __name__ == '__main__':
    main.cli()
