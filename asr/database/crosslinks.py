from typing import Union, List, Dict, Any
from asr.core import (command, option, argument, ASRResult, prepare_result,
                      read_json)
from ase.db import connect
from ase.db.row import AtomsRow
import typing


@prepare_result
class Result(ASRResult):
    """Container for crosslinks results."""
    link_db: str
    connection_dbs: typing.List[str]

    key_descriptions: typing.Dict[str, str] = dict(
        link_db='DB that links get created for.',
        connection_dbs='List of DB that link_db should create links for.'
    )


@command('asr.database.crosslinks',
         returns=Result)
@option('--databaselink', type=str,
        help='DB that links get created for.')
@argument('databases', nargs=-1, type=str,
          help='DBs that target DB should link to.')
def create(databaselink: str,
           databases: Union[str, None] = None) -> Result:
    """Create links between entries in given ASE databases.

    """
    # connect to target and linking DBs
    link_db = connect(databaselink)
    db_connections = {}
    for dbfilenames in databases:
        db = connect(dbfilenames)
        db_connections[dbfilenames] = db

    # make sure all DBs possess the correct metadata structure
    check_metadata(link_db, db_name, target=True)
    for dbfilename, dbconnection in db_connections.items():
        check_metadata(dbconnection, dbfilename, target=False)

    # create dictionary which maps all uids to a row for a given database
    # save the uids_to_row dictionary for every database
    uids_for_each_db = {}
    for dbfilename, dbconnection in db_connections.items():
        uids_to_row = {}
        for row in dbconnection.select(include_data=False):
            uids_to_row[row.uid] = row
        uids_for_each_db[dbfilename] = uids_to_row
    print(uids_for_each_db)

    linkfilename = 'links.json'
    # loop over all rows of the database to link to
    for i, refrow in enumerate(link_db.select()):
        data = refrow.data
        # if links.json present in respecttive row, create links
        if linkfilename in data:
            print('links.json present! Start linking entries...')
            formatted_links = []
            uids_to_link_to = refrow.data[linkfilename]
            for uid in uids_to_link_to['uids']:
                for dbfilename, uids_to_row in uids_for_each_db.items():
                    print(uids_to_row.keys(), uid)
                    metadata = db_connections[dbfilename].metadata
                    row = uids_to_row.get(uid, None)
                    if not row:
                        continue
                    title = metadata['title']
                    link_name_pattern = metadata['link_name']
                    link_url_pattern = metadata['link_url']
                    if row:
                        print('DOING STUFF')
                        name = link_name_pattern.format(row=row, metadata=metadata)
                        url = link_url_pattern.format(row=row, metadata=metadata)
                        formatted_links.append((name, url, title))
            if formatted_links:
                data['links'] = formatted_links
                link_db.update(refrow.id, data={"links": data['links']})

    return Result.fromdata(
        link_db=databaselink,
        connection_dbs=databases)


if __name__ == '__main__':
    main.cli()
