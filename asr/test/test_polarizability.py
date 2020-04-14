import pytest


@pytest.mark.ci
def test_polarizability(separate_folder, mockgpaw, test_material,
                        get_webcontent):
    from asr.polarizability import main
    test_material.write('structure.json')
    main()
    content = get_webcontent()
    assert "polarizability" in content, content
