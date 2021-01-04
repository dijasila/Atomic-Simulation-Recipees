import pytest


@pytest.mark.ci
def test_plasmafrequency(asr_tmpdir_w_params, get_webcontent, mockgpaw,
                         test_material):
    """Test of the plasma freuquency recipe."""
    from asr.plasmafrequency import main
    from pathlib import Path
    if sum(test_material.pbc) != 2:
        pytest.xfail("Plasma frequency is only implemented for 2D atm.")
    main(atoms=test_material)
    assert not Path('es_plasma.gpw').is_file()
    test_material.write('structure.json')
    content = get_webcontent()
    assert "plasmafrequency" in content
