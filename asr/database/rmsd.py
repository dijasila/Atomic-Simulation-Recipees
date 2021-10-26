from typing import Union
from asr.core import command, argument, option, ASRResult
import numpy as np
from datetime import datetime
from asr.utils import timed_print


def normalize_nonpbc_atoms(atoms1, atoms2):
    atoms1, atoms2 = atoms1.copy(), atoms2.copy()

    pbc1_c = atoms1.get_pbc()
    pbc2_c = atoms2.get_pbc()

    assert all(pbc1_c == pbc2_c)

    if not all(pbc1_c):
        cell1_cv = atoms1.get_cell()
        n1_c = (cell1_cv ** 2).sum(1) ** 0.5
        cell2_cv = atoms2.get_cell()
        n2_c = (cell2_cv ** 2).sum(1) ** 0.5
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

    atoms1 = atoms1.copy()
    atoms2 = atoms2.copy()
    atoms1.set_pbc(True)
    atoms2.set_pbc(True)
    niggli_reduce(atoms1)
    niggli_reduce(atoms2)
    struct1 = adaptor.get_structure(atoms1)
    struct2 = adaptor.get_structure(atoms2)

    struct1, struct2 = matcher._process_species([struct1, struct2])
    if not matcher._subset and matcher._comparator.get_hash(
        struct1.composition
    ) != matcher._comparator.get_hash(struct2.composition):
        return None

    struct1, struct2, fu, s1_supercell = matcher._preprocess(struct1, struct2)
    match = matcher._match(
        struct1, struct2, fu, s1_supercell, break_on_match=False, use_rms=False
    )
    if match is None:
        return None
    else:
        rmsd = match[0]
        # Fix normalization
        vol = atoms1.get_volume()
        natoms = len(atoms1)
        old_norm = (natoms / vol) ** (1 / 3)
        rmsd /= old_norm  # Undo
        return rmsd


def update_rmsd(rmsd_by_id, rowid, otherrowid, rmsd):
    if rowid not in rmsd_by_id:
        rmsd_by_id[rowid] = {otherrowid: rmsd}
    else:
        rmsd_by_id[rowid][otherrowid] = rmsd


@command(module="asr.database.rmsd")
@argument("databaseout", required=False, type=str)
@argument("database", type=str)
@option(
    "-c",
    "--comparison-keys",
    help="Keys that have to be identical for RMSD to be calculated.",
    type=str,
)
@option("-r", "--max-rmsd", help="Maximum allowed RMSD.", type=float)
@option(
    "--skip-distance-calc",
    default=False,
    is_flag=True,
    help="Skip distance calculation. Only match structures "
    "based on their reduced formula and comparison_keys.",
)
def main(
    database: str,
    databaseout: Union[str, None] = None,
    comparison_keys: str = "",
    max_rmsd: float = 1.0,
    skip_distance_calc: bool = False,
) -> ASRResult:
    """Calculate RMSD between materials of a database.

    Uses pymatgens StructureMatcher to calculate rmsd. If
    ``databaseout`` is specified a new database will be written to the
    given filename with extra data for rows where a similar row exists
    in ``row.data['results-asr.database.rmsd.json']``. The structure
    of this data is similar to ``rmsd_by_id``. It also stores two
    extra key-value-pairs ``row.min_rmsd`` and ``row.min_rmsd_uid``
    containing the minimum rmsd of the current material to any other
    material and ``uid`` of that other material.

    Note
    ----
    Please note that for systems <3D the computed RMSD can still be
    larger than max_rmsd due to a renormalization of the RMSD measure.
    Normally, a large value is preferred.

    The structure of ``rmsd_by_id`` is::

        {
            '1': {'2': 0.01},
            '2': {'1': 0.01},
        }

    Parameters
    ----------
    database : str
        ASE database filename.
    databaseout : str or None
        If not None, write a new database with rmsd data. Default is None.
    comparison_keys : str
        Comma separated string of keys that should be identical between
        rows to be compared. Eg. 'magstate,natoms'. Default is ''.
    max_rmsd : float
        Maximum rmsd allowed for RMSD to be calculated.
    skip_distance_calc : bool
        If true, only use reduced formula and comparison_keys to match
        structures. Skip calculating distances between structures. The
        output rmsd's will be 0 for matching structures.

    Returns
    -------
    dict
        Keys:
            - ``rmsd_by_id``: RMSDs between materials. The keys are the uids.
            - ``uid_key``: uid_key of the database.

    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.formula import Formula

    # from ase.db import connect
    # db = connect(database)
    adaptor = AseAtomsAdaptor()
    matcher = StructureMatcher(
        primitive_cell=False, attempt_supercell=True, stol=max_rmsd
    )

    comparison_keys = comparison_keys.split(",")

    # Try to figure out what the UID key should be
    row = database.get(id=1)
    uid_key = "uid" if "uid" in row else "id"

    rows = {}
    for row in database.select(include_data=False):
        rows[row.get(uid_key)] = (row.toatoms(), row, Formula(row.formula).reduce()[0])

    print("Calculating RMSDs for all materials...")
    nmat = len(rows)
    rmsd_by_id = {}
    for rowid, (atoms, row, reduced_formula) in rows.items():
        now = datetime.now()
        timed_print(f"{now:%H:%M:%S} {row.id}/{nmat}", wait=30)
        for otherrowid, (otheratoms, otherrow, other_reduced_formula) in rows.items():
            if rowid == otherrowid:
                continue

            if not reduced_formula == other_reduced_formula:
                continue

            # Skip calculation if it has been performed already
            if rowid in rmsd_by_id and otherrowid in rmsd_by_id[rowid]:
                continue

            if comparison_keys and not all(
                row.get(key) == otherrow.get(key) for key in comparison_keys
            ):
                continue

            if not skip_distance_calc:
                rmsd = get_rmsd(atoms, otheratoms, adaptor=adaptor, matcher=matcher)
            else:
                rmsd = 0
            update_rmsd(rmsd_by_id, rowid, otherrowid, rmsd)
            update_rmsd(rmsd_by_id, otherrowid, rowid, rmsd)

    # XXX Fix writing of database with RMSD values
    # if databaseout is not None:
    #     print('Writing to new database...')
    #     with connect(databaseout) as dbwithrmsd:
    #         for row in database.select():
    #             now = datetime.now()
    #             timed_print(f'{now:%H:%M:%S} {row.id}/{nmat}', wait=30)
    #             data = row.data
    #             key_value_pairs = row.key_value_pairs
    #             uid = row.get(uid_key)
    #             if uid in rmsd_by_id:
    #                 rmsd_dict = rmsd_by_id[uid]
    #                 data['results-asr.database.rmsd.json'] = rmsd_dict
    #                 values = [(val, uid) for uid, val in rmsd_dict.items()
    #                           if val is not None]
    #                 if not values:
    #                     continue
    #                 min_rmsd, min_rmsd_uid = min(values)
    #                 key_value_pairs['min_rmsd'] = min_rmsd
    #                 key_value_pairs['min_rmsd_uid'] = min_rmsd_uid
    #             dbwithrmsd.write(row.toatoms(),
    #                              **key_value_pairs, data=row.data)

    #     dbwithrmsd.metadata = database.metadata
    results = {"rmsd_by_id": rmsd_by_id, "uid_key": uid_key}
    return results


if __name__ == "__main__":
    main.cli()
