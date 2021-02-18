import pytest


@pytest.mark.ci
def test_get_wfs(asr_tmpdir, test_material):
    from pathlib import Path
    from ase.io import write
    from asr.gs import calculate, main
    from asr.get_wfs import main as get_wfs

    write('structure.json', test_material)
    calculate(
        calculator={
            "name": "gpaw",
            "kpts": {"density": 2, "gamma": True},
        },
    )

    main()
    results = get_wfs()

    assert Path('wf.0_0.cube').is_file()
    assert results['above_below'] == (None, None)
