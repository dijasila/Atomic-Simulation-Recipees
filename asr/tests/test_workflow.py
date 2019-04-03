from pathlib import Path
from click.testing import CliRunner
from asr.collect import chdir
from ase.build import bulk
from ase.io import write
import json

runner = CliRunner()

folder = Path(__file__).parent / 'Si'

if not folder.is_dir():
    folder.mkdir()

atoms = bulk('Si', crystalstructure='diamond')
write(folder / 'start.json', atoms)

params = {'asr.relax': {'ecut': 100, 'kptdens': 2.0},
          'asr.gs': {'ecut': 100, 'kptdensity': 2.0},
          'asr.dos': {'density': 6.0},
          'asr.borncharges': {'kpointdensity': 1.0}}

Path(folder / 'params.json').write_text(json.dumps(params))


def test_gs():
    with chdir(folder):
        from asr.gs import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0


def test_quickinfo():
    with chdir(folder):
        from asr.quickinfo import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0


def test_relax():
    with chdir(folder):
        from asr.relax import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0


def test_gs_nm():
    with chdir(folder / 'nm'):
        from asr.gs import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0


def test_quickinfo_nm():
    with chdir(folder / 'nm'):
        from asr.quickinfo import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0


def test_borncharges():
    with chdir(folder):
        from asr.borncharges import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0


def test_dos():
    with chdir(folder):
        from asr.dos import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0


def test_collect():
    with chdir(folder):
        from asr.collect import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0


def test_collect_nm():
    with chdir(folder / 'nm'):
        from asr.collect import main
        result = runner.invoke(main, [])

    assert result.exit_code == 0


shutil.rmtree(folder, ignore_errors=False, onerror=None)
