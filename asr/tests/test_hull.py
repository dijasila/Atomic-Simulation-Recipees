import os
from pathlib import Path
from ase.build import bulk
from asr.relax import main as relax
from asr.gs import main as gs
from asr.phonons import main as phonons
from asr.collect import collect
from ase.convex_hull import main as chull
from asr.utils import chdir


def test_cuag():
    structures = [
        bulk('Cu'),
        bulk('Au'),
        bulk('CuAu', crystalstructure='rocksalt', a=4.0),
        bulk('CuAu', crystalstructure='zincblende', a=4.0)]

    os.environ['ASR_TEST_MODE'] = '1'

    for atoms in structures:
        dir = Path(atoms.get_chemical_formula())
        with chdir(dir, create=True):
            atoms.write('structure.json')

            relax(False, ['nm'], 200.0, 2.0, True)

            with chdir('nm'):
                gs()
                phonons()

    refs = ['Cu/nm/', 'Au/nm']
    alloys = ['Cu/nm/', 'Au/nm']
    collect(refs, 'refs.db')
    collect(alloys, 'alloys.db')

    for dir in alloys:
        with chdir(dir):
            chull()


if __name__ == '__main__':
    test_cuag()
