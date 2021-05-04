import pytest


@pytest.mark.ci
def test_setup_materials(asr_tmpdir_w_params):
    from asr.setup.materials import main
    from pathlib import Path
    main()

    assert Path('materials.json').is_file()
