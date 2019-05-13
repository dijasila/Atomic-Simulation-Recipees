import os
from pathlib import Path

from ase.build import bulk

from asr.collect import main as collect
from asr.convex_hull import main as chull
from asr.gs import main as gs
from asr.phonons import main as phonons
from asr.quickinfo import main as quickinfo
from asr.relax import main as relax
from asr.utils import chdir
import pytest


# We create temporary directory and move the start.json
# and params.json into that directory
@pytest.fixture(scope='class')
def directory(tmpdir_factory):
    path = tmpdir_factory.mktemp('emt')
    return path


def test_cuag(directory):
    with chdir(directory):
        structures = [
            bulk('Cu'),
            bulk('Au'),
            bulk('CuAu', crystalstructure='rocksalt', a=5.0),
            bulk('CuAuAu', crystalstructure='fluorite', a=5.8)]

        os.environ['ASR_TEST_MODE'] = '1'

        for atoms in structures:
            dir = Path(atoms.get_chemical_formula())
            with chdir(dir, create=True, empty=True):
                atoms.write('start.json')

                relax(args=[])

                with chdir('nm'):
                    gs(args=[])
                    phonons(args=[])

        for dir in Path().glob('*u/nm/'):
            with chdir(dir):
                quickinfo(args=[])

        db = Path('database.db')
        if db.is_file():
            db.unlink()

        collect(args=[str(dir) for dir in Path().glob('?u/nm/')])
        db.rename('refs.db')

        collect(args=[str(dir) for dir in Path().glob('Au*Cu/nm/')])
        db.rename('database1.db')

        for dir in Path().glob('Au*Cu/nm/'):
            with chdir(dir):
                chull(args=['-r', '../../refs.db',
                            '-d', '../../database1.db'])

        collect(args=[str(dir) for dir in Path().glob('Au*Cu/nm/')])
