from typing import Union
from asr.core import command, option, argument
from ase.db import connect


@command('asr.database.crosslinks')
@argument('databases', nargs=-1, type=str)
def main(databases: Union[str, None] = None):
    """
    Create links between entries in given ASE databases.
    """
    dblist = []
    for element in databases:
        db = connect(element)
        dblist.append(db)
        print(db.metadata)

    links = {'link_db': {}}
    print(f"INFO: create links for webpanel of DB {dblist[0].metadata['title']}")
    print(f"INFO: link to the following databases:")
    for i in range(1, len(dblist)):
        print(f"..... {dblist[i].metadata['title']}")
    for database in dblist:
        print(f"INFO: creating links to database {database.metadata['title']}")
        for i, refrow in enumerate(dblist[0].select()):
            linklist = []
            urllist = []
            data = {}
            for j, row in enumerate(database.select()):
                if row.link_uid == refrow.link_uid:
                    name = database.metadata['internal_links']['link_name']
                    url = database.metadata['internal_links']['link_url']
                    link_name = eval(f"f'{name}'")
                    link_url = eval(f"f'{url}'")
                    linklist.append(link_name)
                    urllist.append(link_url)
            data['links'] = {f"{database.metadata['title']}": {'link_names': linklist,
                             'link_urls': urllist}}
            print(data)
        print('INFO: DONE!')


    return None
