"""Run integrity checks of database."""

from ase.db import connect


def main(dbname: str):
    """Run a check of a database.

    Check whether all child uids exists and whether there are any
    duplicate uids.

    """
    with connect(dbname, serial=True) as db:
        uids = set()
        missing_uids = set()
        duplicate_uids = set()
        child_uids = set()
        for row in db.select():
            if "uid" not in row:
                missing_uids.add(row.id)
            else:
                if row.uid in uids:
                    duplicate_uids.add(row.uid)
                else:
                    uids.add(row.uid)
            children = row.data.get("__children__", {})
            child_uids.update(set(children.values()))

    missing_child_uids = child_uids - uids
    nmissing_childs = len(missing_child_uids)

    if missing_child_uids:
        print(f"Missing {nmissing_childs} child uids in database. " "Child uids:")
        for uid in missing_child_uids:
            print(" " * 4, uid)

    nduplicates = len(duplicate_uids)
    if duplicate_uids:
        print(f"{nduplicates} duplicate uids in database. " "Duplicate uids:")
        for uid in duplicate_uids:
            print(" " * 4, uid)

    nmissing_uids = len(missing_uids)
    if missing_uids:
        print(f"{nmissing_uids} missing uids in database. " "Missing row.id's:")
        for uid in missing_uids:
            print(" " * 4, uid)

    return {
        "missing_child_uids": missing_child_uids,
        "duplicate_uids": duplicate_uids,
        "missing_uids": missing_uids,
    }
