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


@command('asr.database.crosslinks')
@option('--databaselink', type=str)
@argument('databases', nargs=-1, type=str)
def create(databaselink: str,
           databases: Union[str, None] = None) -> Result:
    """Create links between entries in given ASE databases."""
    # connect to the link database and create dictionary of databases to
    # link to
    link_db = connect(databaselink)
    db_connections = {}  # [link_db]
    for dbfilenames in databases:
        db = connect(dbfilenames)
        db_connections[dbfilenames] = db

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


# @prepare_result
# class Result(ASRResult):
#     """Container for database crosslinks results."""
# 
#     linked_database: str
#     links: typing.List[LinkResults]
# 
#     key_descriptions = dict(
#         linked_database='Database that crosslinks got created for.',
#         links='List of LinkResults containers.')
# 
 
# def return_link_results(linklist, urllist, db_name) -> LinkResults:
#     return LinkResults.fromdata(
#         name=db_name,
#         link_names=linklist,
#         link_urls=urllist)
# 
# 
# @command(module='asr.database.crosslinks',
#          dependencies=['asr.database.material_fingerprint',
#                        'asr.database.crosslinks@create'],
#          returns=Result)
# @argument('database', nargs=1, type=str)
# def main(database: str) -> Result:
#     """Create links.
# 
#     Use created crosslink names and urls from asr.database.crosslinks@create
#     and write HTML code for representation on webpage.
#     """
#     # First, get uid of structure in current directory to compare to respective
#     # uid in the database
#     results_fingerprint = read_json('results-asr.database.material_fingerprint.json')
#     structure_uid = results_fingerprint['uid']
# 
#     # Second, connect to the crosslinked database and obtain the names, urls,
#     # and types
#     db = connect(database)
#     for row in db.select():
#         if row.uid == structure_uid:
#             links = row.data['links']
# 
#     link_results_list = []
#     for element in links:
#         link_results = return_link_results(links[element]['link_names'],
#                                            links[element]['link_urls'],
#                                            element)
#         link_results_list.append(link_results)
# 
#     return Result.fromdata(linked_database=database,
#                            links=link_results_list)
# 
# 
# def link_tables(row: AtomsRow) -> List[Dict[str, Any]]:
#     import numpy as np
# 
#     data = row.data.get('results-asr.database.crosslinks.json')
#     links = data['links']
#     names = []
#     urls = []
#     types = []
#     for element in links:
#         for j in range(len(links[element]['link_names'])):
#             names.append(links[element]['link_names'][j])
#             urls.append(links[element]['link_urls'][j])
#             types.append(element)
# 
#     table_array = np.array([names, urls, types])
# 
#     return table_array


if __name__ == '__main__':
    main.cli()
