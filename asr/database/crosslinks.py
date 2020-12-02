from typing import Union, List, Dict, Any
from asr.core import (command, option, argument, ASRResult, prepare_result,
                      read_json)
from ase.db import connect
from ase.db.row import AtomsRow
import typing


# TODO: add main function to create webpanel
# TODO: clean up
# TODO: exclude linking to itself
# TODO: test new results implementation
# TODO: webpanel not changed to working one yet!


@command('asr.database.crosslinks')
@option('--databaselink', type=str)
@argument('databases', nargs=-1, type=str)
def create(databaselink: str,
           databases: Union[str, None] = None):
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

    # print(f"INFO: create links for webpanel of DB {link_db.metadata['title']}")
    # print(f"INFO: link to the following databases:")
    # for i in range(0, len(db_connections)):
    #     print(f"..... {db_connections[i].metadata['title']}")

    linkfilename = 'links.json'
    # loop over all rows of the database to link to
    for i, refrow in enumerate(link_db.select()):
        data = refrow.data
        # if links.json present in respecttive row, create links
        if linkfilename in data:
            formatted_links = []
            uids_to_link_to = refrow.data[linkfilename]
            for uid in uids_to_link_to:
                for dbfilename, uids_to_row in uids_for_each_db.items():
                    metadata = db_connections[dbfilename].metadata
                    row = uids_to_row.get(uid, None)
                    if not row:
                        continue
                    title = metadata['title']
                    link_name_pattern = metadata['link_name']
                    link_url_pattern = metadata['link_url']
                    if row:
                        name = link_name_pattern.format(row=row, metadata=metadata)
                        url = link_url_pattern.format(row=row, metadata=metadata)
                        formatted_links.append((name, url, title))
            if formatted_links:
                data['links'] = formatted_links
                link_db.update(refrow.id, data={"links": data['links']})

        # data = {'links': {}}
        # refid = refrow.id
        # for database in db_connections:
        #     linklist = []
        #     urllist = []
        #     for j, row in enumerate(database.select()):
        #         if row.link_uid == refrow.link_uid:
        #             name = database.metadata['internal_links']['link_name']
        #             url = database.metadata['internal_links']['link_url']
        #             link_name = '{name}'.format(name=name)
        #             link_url = eval(f"f'{url}'")
        #             linklist.append(link_name)
        #             urllist.append(link_url)
        #     data['links'][f"{database.metadata['title']}"] = {'link_names': linklist,
        #                                                       'link_urls': urllist}
        # link_db.update(refid, data={"links": data['links']})


@prepare_result
class LinkResults(ASRResult):
    """Container for links to a specific Database."""

    name: str
    link_names: typing.List[str]
    link_urls: typing.List[str]

    key_descriptions = dict(
        name='Name of the DB that that the initial DB is linked to.',
        link_names='List of names of the links to that specific DB.',
        link_urls='List of urls of the links to that specific DB.')


@prepare_result
class Result(ASRResult):
    """Container for database crosslinks results."""

    linked_database: str
    links: typing.List[LinkResults]

    key_descriptions = dict(
        linked_database='Database that crosslinks got created for.',
        links='List of LinkResults containers.')


def return_link_results(linklist, urllist, db_name) -> LinkResults:
    return LinkResults.fromdata(
        name=db_name,
        link_names=linklist,
        link_urls=urllist)


@command(module='asr.database.crosslinks',
         dependencies=['asr.database.material_fingerprint',
                       'asr.database.crosslinks@create'],
         returns=Result)
@argument('database', nargs=1, type=str)
def main(database: str) -> Result:
    """Create links.

    Use created crosslink names and urls from asr.database.crosslinks@create
    and write HTML code for representation on webpage.
    """
    # First, get uid of structure in current directory to compare to respective
    # uid in the database
    results_fingerprint = read_json('results-asr.database.material_fingerprint.json')
    structure_uid = results_fingerprint['uid']

    # Second, connect to the crosslinked database and obtain the names, urls,
    # and types
    db = connect(database)
    for row in db.select():
        if row.uid == structure_uid:
            links = row.data['links']

    link_results_list = []
    for element in links:
        link_results = return_link_results(links[element]['link_names'],
                                           links[element]['link_urls'],
                                           element)
        link_results_list.append(link_results)

    return Result.fromdata(linked_database=database,
                           links=link_results_list)


def link_tables(row: AtomsRow) -> List[Dict[str, Any]]:
    import numpy as np

    data = row.data.get('results-asr.database.crosslinks.json')
    links = data['links']
    names = []
    urls = []
    types = []
    for element in links:
        for j in range(len(links[element]['link_names'])):
            names.append(links[element]['link_names'][j])
            urls.append(links[element]['link_urls'][j])
            types.append(element)

    table_array = np.array([names, urls, types])

    return table_array


if __name__ == '__main__':
    main.cli()
