import pytest


@pytest.mark.ci
def test_setup_params(separate_folder):
    from asr.core import read_json
    from asr.setup.scanparams import main
    from pathlib import Path
    params = [3, 4, 5]
    recipe = 'asr.phonons@calculate'
    key = 'kptdensity'
    main(scanparams=[f'{recipe}:{key}'] + list(map(str, params)))

    for i, param in enumerate(params):
        p = Path(f'scanparams{i}', 'params.json')
        assert p.is_file()
        params = read_json(p)
        assert params[recipe][key] == param
