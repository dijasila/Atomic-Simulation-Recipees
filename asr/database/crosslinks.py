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

    for i, row in enumerate(dblist[0].select()):
        for j, row2 in enumerate(dblist[1].select()):
            if row.link_uid == row2.link_uid:
                id = j
                # dblist[0].update(j, ...)
                name = dblist[0].metadata['internal_links']['link_name']
                print(name)


    return None
