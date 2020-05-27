import pytest

def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, test_material):
    from asr.gs import main

    test_material.write('structure.json')
    results = main()
    assert results['gap'] == pytest.approx(0)
    main()
