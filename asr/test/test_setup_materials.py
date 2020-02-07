import pytest


@pytest.mark.ci
def test_setup_magnetize(separate_folder):
    from asr.setup.materials import main
    from ase.db import connect
    from pathlib import Path
    main()

    assert Path('materials.json').is_file()
