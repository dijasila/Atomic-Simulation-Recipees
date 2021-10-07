"""Merge ASE databases."""


def main(databases: str, databaseout: str, identifier: str = "uid"):
    """Merge two ASE databases."""
    from ase.db import connect
    from pathlib import Path
    from click import progressbar

    print(f"Merging {databases} into {databaseout}")

    def item_show_func(item):
        if item is None:
            return item
        return str(item.formula)

    dest = Path(databaseout)
    assert not dest.is_file(), f"The destination path {databaseout} already exists."

    # We build up a temporary database file at this destination
    tmpdest = Path(databaseout + ".tmp.db")
    if tmpdest.is_file():
        tmpdest.unlink()

    # First merge rows common in both databases
    metadata = {"keys": set()}
    for database in databases:
        # Database for looking up existing materials
        tmpdestsearch = Path("_" + str(tmpdest))
        if tmpdestsearch.is_file():
            tmpdestsearch.unlink()

        if tmpdest.is_file():
            tmpdest.rename(tmpdestsearch)

        print(f"Connecting to {database} (may take a while)", flush=True)
        db = connect(database)
        dbsearch = connect(str(tmpdestsearch))
        dbmerged = connect(str(tmpdest))

        with dbmerged, progressbar(
            db.select(),
            length=len(db),
            label=f"Merging {database}",
            item_show_func=item_show_func,
        ) as selection:
            id_matches = []
            for row1 in selection:
                structid = row1.get(identifier)
                matching = list(dbsearch.select(f"{identifier}={structid}"))

                if len(matching) > 1:
                    raise RuntimeError(
                        "More than one structure "
                        f"in {databaseout} "
                        f"matching {identifier}={structid}"
                    )
                elif len(matching) == 0:
                    atoms = row1.toatoms()
                    kvp = row1.key_value_pairs
                    data = row1.data
                else:
                    row2 = matching[0]
                    id_matches.append(row2.id)
                    data = row2.data
                    kvp = row2.key_value_pairs

                    data.update(row1.data.copy())
                    kvp.update(row1.key_value_pairs.copy())

                    atoms = row1.toatoms()
                    atoms2 = row2.toatoms()
                    assert atoms == atoms2, "Atoms not matching!"

                dbmerged.write(atoms, key_value_pairs=kvp, data=data)

            # Write the remaining rows from db2 that wasn't matched
            for row2 in dbsearch.select():
                if row2.id not in id_matches:
                    dbmerged.write(
                        row2.toatoms(),
                        key_value_pairs=row2.key_value_pairs,
                        data=row2.data,
                    )

        # Update metadata
        metadata["keys"].update(set(db.metadata.get("keys", [])))

    metadata["keys"] = sorted(list(metadata["keys"]))
    dbmerged.metadata = metadata

    # Remove lookup db
    tmpdestsearch.unlink()

    # Copy the file to the final destination
    tmpdest.rename(dest)
