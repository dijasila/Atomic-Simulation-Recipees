from asr.core import command, argument, option


@command('asr.database.merge')
@argument('databaseout', nargs=1)
@argument('databases', nargs=-1)
@option('--identifier', help='Identifier for matching database entries.')
def main(databases, databaseout, identifier='asr_id'):
    """Merge two ASE databases."""
    from ase.db import connect
    from pathlib import Path

    print(f'Merging {databases} into {databaseout}')

    def item_show_func(item):
        if item is None:
            return item
        return str(item.formula)

    dest = Path(databaseout)
    assert not dest.is_file(), \
        f'The destination path {databaseout} already exists.'

    # We build up a temporary database file at this destination
    tmpdest = Path(databaseout + '.tmp.db')
    if tmpdest.is_file():
        tmpdest.unlink()

    # First merge rows common in both databases
    for i_d, database in enumerate(databases):
        # Database for looking up existing materials
        tmpdestsearch = Path('_' + str(tmpdest))
        if tmpdestsearch.is_file():
            tmpdestsearch.unlink()

        if tmpdest.is_file():
            tmpdest.rename(tmpdestsearch)

        db = connect(database)
        dbsearch = connect(str(tmpdestsearch))
        dbmerged = connect(str(tmpdest))

        # Update metadata
        metadata = {}
        if dbsearch.metadata:
            metadata.update(dbsearch.metadata.copy())
        if db.metadata:
            metadata.update(db.metadata.copy())
        dbmerged.metadata = metadata

        selection = db.select()
        with dbmerged:
            id_matches = []
            for row1 in selection:
                structid = row1.get(identifier)
                matching = list(dbsearch.select(f'{identifier}={structid}'))

                if len(matching) > 1:
                    raise RuntimeError('More than one structure '
                                       f'in {databaseout} '
                                       f'matching {identifier}={structid}')
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

                    atoms1 = row1.toatoms()
                    atoms2 = row2.toatoms()
                    assert atoms1 == atoms2, 'Atoms not matching!'
                    atoms = atoms1

                dbmerged.write(atoms,
                               key_value_pairs=kvp,
                               data=data)

            # Write the remaining rows from db2 that wasn't matched
            for row2 in dbsearch.select():
                if row2.id not in id_matches:
                    dbmerged.write(row2.toatoms(),
                                   key_value_pairs=row2.key_value_pairs,
                                   data=row2.data)

    # Remove lookup db
    tmpdestsearch.unlink()

    # Copy the file to the final destination
    tmpdest.rename(dest)


if __name__ == '__main__':
    main.cli()
