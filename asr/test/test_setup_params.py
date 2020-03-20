import pytest
from asr.core import get_recipes

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


@pytest.mark.ci
def test_setup_params_input_dict(separate_folder):
    """Test that setup.params works with an input dict."""
    from asr.setup.params import main
    from asr.core import read_json
    params = {'asr.gs@calculate': {'calculator': {'name': 'testname', None: None}}}
    main(params=params)
    params = read_json('params.json')
    assert params['asr.gs@calculate']['calculator']['name'] == 'testname'
    assert params['asr.gs@calculate']['calculator']['charge'] == 0


@pytest.mark.ci
def test_setup_params_recurse_dict(separate_folder):
    from asr.setup.params import main
    from asr.core import read_json
    main(params=['asr.gs@calculate:calculator',
                 '{"name": "testname", "mode": {"ecut": 400, ...}, ...}'])

    params = read_json('params.json')
    assert params['asr.gs@calculate']['calculator']['name'] == 'testname'
    assert params['asr.gs@calculate']['calculator']['mode']['name'] == 'pw'
    assert params['asr.gs@calculate']['calculator']['mode']['ecut'] == 400


recipes = get_recipes()


@pytest.mark.ci
@pytest.mark.parametrize("recipe",
                         list(filter(lambda x: x.name != 'asr.setup.params',
                                     recipes)))
def test_setup_params_parametrize(separate_folder, recipe):
    from asr.setup.params import main as setupparams
    defparams = recipe.defparams
    defparamdict = {recipe.name: defparams}
    setupparams(params=defparamdict)
    setupparams(params=defparamdict)
