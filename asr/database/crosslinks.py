from typing import Union, List, Dict, Any
from asr.core import (command, option, argument, ASRResult, prepare_result,
                      read_json)
from ase.db import connect
from ase.db.row import AtomsRow
import typing


# TODO: add main function to create webpanel
# TODO: clean up
# TODO: exclude linking to itself


def webpanel(result, row, key_descriptions):
    """Creates a webpanel containing all of the links that got created with
    asr.database.crosslinks@create."""
    from asr.database.browser import table

    table_array = link_tables(row)
    print(table_array)

    caption = ""
    link_table = table(row,
                       '',
                       ['link', 'linktype'],
                       key_descriptions)
    panel = {'title': 'Internal and external crosslinks',
             'columns': [table_array],
             'sort': 1}
    summary = {'title': 'Summary',
               'columns': [[{'type': 'table',
                             'header': [''],
                             'rows': [row],
                             'columnwidth': 3}]],
               'sort': 1}

    return [panel, summary]


@command('asr.database.crosslinks')
@option('--databaselink', type=str)
@argument('databases', nargs=-1, type=str)
def create(databaselink: str,
           databases: Union[str, None] = None):
    """
    Create links between entries in given ASE databases.
    """
    link_db = connect(databaselink)
    dblist = [link_db]
    for element in databases:
        db = connect(element)
        dblist.append(db)

    print(f"INFO: create links for webpanel of DB {link_db.metadata['title']}")
    print(f"INFO: link to the following databases:")
    for i in range(0, len(dblist)):
        print(f"..... {dblist[i].metadata['title']}")

    for i, refrow in enumerate(link_db.select()):
        data = {'links': {}}
        refid = refrow.id
        for database in dblist:
            linklist = []
            urllist = []
            for j, row in enumerate(database.select()):
                if row.link_uid == refrow.link_uid:
                    name = database.metadata['internal_links']['link_name']
                    url = database.metadata['internal_links']['link_url']
                    link_name = eval(f"f'{name}'")
                    link_url = eval(f"f'{url}'")
                    linklist.append(link_name)
                    urllist.append(link_url)
            data['links'][f"{database.metadata['title']}"] = {'link_names': linklist,
                                                              'link_urls': urllist}
        link_db.update(refid, data={"links": data['links']})


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

    formats = {"ase_webpanel": webpanel}


@command(module='asr.database.crosslinks',
         requires=['asr.database.material_fingerprint',
                   'asr.database.crosslinks@create'],
         dependencies=['asr.database.material_fingerprint',
                       'asr.database.crosslinks@create'],
         returns=Result)
@argument('database', nargs=1, type=str)
def main(database: str) -> Result:
    """Use created crosslink names and urls from asr.database.crosslinks@create
    and write HTML code for representation on webpage."""
    # First, get uid of structure in current directory to compare to respective
    # uid in the database
    results_fingerprint = read_json('results-asr.database.material_fingerprint.json')
    structure_uid = results_fingerprint['uid']
    print(structure_uid)

    # Second, connect to the crosslinked database and obtain the names, urls,
    # and types
    db = connect(database)
    for row in db.select():
        if row.uid == structure_uid:
            links = row.data['links']

    return Result.fromdata(linked_database=database,
                           links=links)


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
