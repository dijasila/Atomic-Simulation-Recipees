from typing import Union, List, Dict, Any
from asr.core import command, option, argument, ASRResult, prepare_result, write_json
from ase.db import connect
from ase.db.row import AtomsRow


# TODO: add main function to create webpanel
# TODO: clean up
# TODO: exclude linking to itself

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
class Result(ASRResult):
    """Container for database crosslinks results."""
    linked_database: str

    key_descriptions = dict(
        linked_database='Database that crosslinks got created for.')

    # formats = {"ase_webpanel": webpanel}


@command(module='asr.database.crosslinks',
         dependencies=['asr.database.crosslinks@create'],
         returns=Result)
@argument('database', nargs=1, type=str)
def main(database: str) -> Result:
    """Use created crosslink names and urls from asr.database.crosslinks@create
    and write HTML code for representation on webpage."""
    db = connect(database)
    for row in db.select():
        link_tables(row)

    return Result.fromdata(linked_database=database)


def webpanel(result, row, key_descriptions):
    """Creates a webpanel containing all of the links that got created with
    asr.database.crosslinks@create."""
    return None


def link_tables(row: AtomsRow) -> List[Dict[str, Any]]:
    data = row.data['links']
    for element in data:
        names = row.data['links'][element]['link_names']
        urls = row.data['links'][element]['link_urls']
        types = element
        print(names, urls, types)


if __name__ == '__main__':
    main.cli()
