from asr.core import command, argument, option


def folderexists():
    from pathlib import Path
    assert Path('tree').is_dir()


tests = [
    {'cli': ['asr run setup.materials',
             'asr run database.totree materials.json --run'],
     'test': folderexists}
]


@command('asr.database.totree',
         tests=tests)
@argument('databaseout', nargs=1)
@argument('databases', nargs=-1)
@option('--identifier', help='Identifier for matching database entries.')
def main(databases, databaseout, identifier='asr_id'):
    """Merge two ASE databases."""
    from ase.db import connect
    from pathlib import Path
    from click import progressbar
    dbmerged = connect(databaseout)

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
    for database in databases:
        # Database for looking up existing materials
        tmpdestsearch = Path('_' + str(tmpdest))
        if tmpdestsearch.is_file():
            tmpdestsearch.unlink()

        if tmpdest.is_file():
            tmpdest.rename(tmpdestsearch)

        db = connect(database)
        dbsearch = connect(str(tmpdestsearch))
        dbmerged = connect(str(tmpdest))

        selection = progressbar(list(db.select()),
                                label=f'Merging {database}',
                                item_show_func=item_show_func)
        with dbmerged, dbsearch, selection:
            for row1 in selection:
                structid = row1.get(identifier)
                matching = list(dbsearch.select(f'{identifier}={structid}'))

                if len(matching) > 1:
                    raise RuntimeError('More than one structure '
                                       f'in {databaseout} '
                                       f'matching identifier={identifier}')
                elif len(matching) == 0:
                    dbmerged.write(row1.toatoms(),
                                   key_value_pairs=row1.key_value_pairs,
                                   data=row1.data)
                else:
                    row2 = matching[0]
                    data = row2.data.copy()
                    kvp = row2.key_value_pairs.copy()

                    data.update(row1.data)
                    kvp.update(row1.key_value_pairs)

                    atoms1 = row1.toatoms()
                    atoms2 = row2.toatoms()
                    assert atoms1 == atoms2, 'Atoms not matching!'
                    dbmerged.write(row1.toatoms(),
                                   data=data,
                                   key_value_pairs=kvp)

    # Remove lookup db
    tmpdestsearch.unlink()

    # Copy the file to the final destination
    tmpdest.rename(dest)


if __name__ == '__main__':
    main.cli()
