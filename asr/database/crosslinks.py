from typing import Union
from asr.core import command, option, argument


@command('asr.database.crosslinks')
@argument('databases', nargs=-1, type=str)
def main(databases: Union[str, None] = None):
    """
    Create links between entries in given ASE databases.
    """

    return None
