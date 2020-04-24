import pytest


@pytest.mark.ci
def test_plasmafrequency(asr_tmpdir_w_params, get_webcontent, mockgpaw,
                         test_material):
    """Test of the plasma freuquency recipe."""
    from asr.plasmafrequency import calculate, main
    from pathlib import Path
    if sum(test_material.pbc) != 2:
        pytest.xfail("Plasma frequency is only implemented for 2D atm.")
    test_material.write('structure.json')
    calculate()
    assert Path('es_plasma.gpw').is_file()
    main()
    assert not Path('es_plasma.gpw').is_file()
    content = get_webcontent()
    assert "plasmafrequency" in content
