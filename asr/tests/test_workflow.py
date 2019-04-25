import shutil
from pathlib import Path
from click.testing import CliRunner
from asr.collect import chdir
from ase.build import bulk
from ase.io import write
import json

runner = CliRunner()

# Set up folder, start.json and params.json
folder = (Path(__file__).parent / 'Si').resolve()
if folder.is_dir():
    shutil.rmtree(folder, ignore_errors=False, onerror=None)
folder.mkdir()

atoms = bulk('Si', crystalstructure='diamond')
write(folder / 'start.json', atoms)
params = {'asr.relax': {'ecut': 100, 'kptdens': 2.0},
          'asr.gs': {'ecut': 100, 'kptdensity': 2.0},
          'asr.dos': {'density': 6.0},
          'asr.borncharges': {'kpointdensity': 1.0},
          'asr.bandstructure': {'npoints': 50},
          'asr.polarizability': {'density': 6.0,
                                 'ecut': 10.0,
                                 'bandfactor': 2},
          'asr.pdos': {'kptdensity': 6.0}}
Path(folder / 'params.json').write_text(json.dumps(params))


def test_workflow():
    with chdir(folder):
        from asr.workflow import main
        main()
        # result = runner.invoke(main, [])

    # assert result.exit_code == 0


if __name__ == '__main__':
    test_workflow()
