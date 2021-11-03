from ase.db import connect
from asr.database import parse_row_data
db = connect('database.db')

all_records = []
for row in db.select('has_asr_bandstructure'):
    records = parse_row_data(row.data["records"])
    all_records.extend(records)
