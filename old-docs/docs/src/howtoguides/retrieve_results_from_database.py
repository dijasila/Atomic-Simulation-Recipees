from ase.db import connect
from asr.database import parse_row_data
db = connect('database.db')

bandstructure_results = []
for row in db.select('has_asr_bandstructure'):
    data = parse_row_data(row.data)
    bs = data['results-asr.bandstructure.json']
    bandstructure_results.append(bs)
