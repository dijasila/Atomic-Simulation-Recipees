from pathlib import Path
from ase.build import bulk
from asr.collect import main as collect
from asr.convex_hull import main as chull
from asr.gs import main as gs
from asr.phonons import main as phonons
from asr.structureinfo import main as structureinfo
from asr.relax import main as relax
from asr.utils import chdir, write_json

params = {'_calculator': 'EMT'}
write_json('params.json', params)

structures = [
    bulk('Cu'),
    bulk('Au'),
    bulk('CuAu', crystalstructure='rocksalt', a=5.0),
    bulk('CuAuAu', crystalstructure='fluorite', a=5.8)]

for atoms in structures:
    dir = Path(atoms.get_chemical_formula())
    with chdir(dir, create=True, empty=True):
        atoms.write('unrelaxed.json')

        relax(args=['--nod3'])

        gs(args=[])
        phonons(args=[])

for dir in Path().glob('*u/'):
    with chdir(dir):
        structureinfo(args=[])

db = Path('database.db')
if db.is_file():
    db.unlink()

collect(args=[str(dir) for dir in Path().glob('?u/')])
db.rename('refs.db')

collect(args=[str(dir) for dir in Path().glob('Au*Cu/')])
db.rename('database1.db')

for dir in Path().glob('Au*Cu/'):
    with chdir(dir):
        chull(args=['-r', '../refs.db',
                    '-d', '../database1.db'])

collect(args=[str(dir) for dir in Path().glob('Au*Cu/')])
