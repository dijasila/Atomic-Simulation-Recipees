import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from ase.db.row import AtomsRow
from ase.db import connect


db_path = '/home/tara/website_backend/database.db'

"""
We need to physically look at each entry in the database to create our 
dataclass. These functions facilitate such a task.

Since the entries from the database are only instatiated from an AtomsRow into
a results object with a bunch of bloat information it is easier to just 
directly read the important data from the dictionary.

We are creating a SEPARATE dataclass to store what has become the windows bloat

"""
def get_atomsrow(data_path):
    row_iter = connect(data_path).select()
    atoms_row = next(row_iter)
    return atoms_row


atoms_row = get_atomsrow(db_path)
for key in atoms_row.data:
    print(key)
    print('\t', atoms_row.data[key].keys())


for entry in atoms_row.data[key]['kwargs']['data']['bs_soc']:
    print(entry)
    print(type(atoms_row.data[key]['kwargs']['data']['bs_soc'][entry]))


for entry in atoms_row.data[key]['kwargs']['data']['bs_soc']:
    soc = atoms_row.data[key]['kwargs']['data']['bs_soc']
    nosoc = atoms_row.data[key]['kwargs']['data']['bs_nosoc']
    print(entry)
    print(soc[entry], nosoc[entry])
    are_equivalent = soc[entry] == nosoc[entry]
    print(are_equivalent)


for entry in atoms_row.data[key]['kwargs']['data']['bs_nosoc']:
    print(type(entry))


# everything stored under kwargs must now be a dataclass - not a dictionary
atoms_row.data[key].keys()
atoms_row.data[key]['kwargs'].keys()
atoms_row.data[key]['kwargs']['data']

