import pytest


@pytest.mark.ci
def test_setup_params(separate_folder):
    from asr.setup.params import main
    from asr.core import read_json
    from pathlib import Path
    main(params=['asr.relax:d3', 'True'])

    p = Path('params.json')
    assert p.is_file()
    params = read_json('params.json')
    assert params['asr.relax']['d3'] is True

    main(params=['asr.gs@calculate:calculator', '{"name": "testname", ...}'])
    params = read_json('params.json')
    assert params['asr.relax']['d3'] is True
    assert params['asr.gs@calculate']['calculator']['name'] == 'testname'
    assert params['asr.gs@calculate']['calculator']['charge'] == 0

    main(params=['asr.relax:d3', 'False'])
    params = read_json('params.json')
    assert params['asr.relax']['d3'] is False
    assert params['asr.gs@calculate']['calculator']['name'] == 'testname'
    assert params['asr.gs@calculate']['calculator']['charge'] == 0

    main(params=['*:kptdensity', '12'])
    params = read_json('params.json')
    assert params["asr.polarizability"]["kptdensity"] == 12
    for value in params.values():
        if 'kptdensity' in value:
            assert value['kptdensity'] == 12
