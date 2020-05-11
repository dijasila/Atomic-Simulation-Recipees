from asr.core import command, argument, option
import numpy as np


def normalize_nonpbc_atoms(atoms1, atoms2):
    atoms1, atoms2 = atoms1.copy(), atoms2.copy()

    pbc1_c = atoms1.get_pbc()
    pbc2_c = atoms2.get_pbc()

    assert all(pbc1_c == pbc2_c)

    if not all(pbc1_c):
        cell1_cv = atoms1.get_cell()
        n1_c = (cell1_cv**2).sum(1)**0.5
        cell2_cv = atoms2.get_cell()
        n2_c = (cell2_cv**2).sum(1)**0.5
        cell2_cv[~pbc2_c] *= (n1_c / n2_c)[~pbc2_c, np.newaxis]
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
@argument('databaseout')
@argument('database')
@option('-r', '--rmsd-tol', help='RMSD tolerance.')
def main(database, databaseout, rmsd_tol=0.3):
    """Take an input database filter out duplicates.

    Uses asr.duplicates.check_duplicates.

    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.formula import Formula
    from ase.db import connect
    db = connect(database)
    adaptor = AseAtomsAdaptor()
    matcher = StructureMatcher(primitive_cell=False,
                               attempt_supercell=True,
                               stol=rmsd_tol)

    row = db.get(id=1)
    uid_key = 'uid' if 'uid' in row else 'id'

    rows = {}
    for row in db.select(include_data=False):
        rows[row.get(uid_key)] = (row.toatoms(), row)

    rmsd_by_id = {}
    for rowid, (atoms, row) in rows.items():
        row_rmsd_by_id = {}
        formula = Formula(row.formula).reduce()[0]
        for otherrowid, (otheratoms, otherrow) in rows.items():
            if rowid == otherrowid:
                continue
            otherformula = Formula(otherrow.formula).reduce()[0]
            if not formula == otherformula:
                continue
            rmsd = get_rmsd(atoms, otherrow.toatoms(),
                            adaptor=adaptor,
                            matcher=matcher)
            if rmsd is None:
                continue
            row_rmsd_by_id[otherrowid] = rmsd
        if row_rmsd_by_id:
            rmsd_by_id[rowid] = row_rmsd_by_id

    with connect(databaseout) as dbwithrmsd:
        for row in db.select():
            data = row.data
            key_value_pairs = row.key_value_pairs
            uid = row.get(uid_key)
            if uid in rmsd_by_id:
                rmsd_dict = rmsd_by_id[uid]
                data['results-asr.database.rmsd.json'] = rmsd_dict
                min_rmsd, min_rmsd_uid = \
                    min((val, uid) for uid, val in rmsd_dict.items())
                key_value_pairs['min_rmsd'] = min_rmsd
                key_value_pairs['min_rmsd_uid'] = min_rmsd_uid
            dbwithrmsd.write(row.toatoms(), **key_value_pairs, data=row.data)

    dbwithrmsd.metadata = db.metadata
    return rmsd_by_id


if __name__ == '__main__':
    main.cli()
