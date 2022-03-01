import pytest
from click.testing import CliRunner


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
def test_setup_params(asr_tmpdir):
    from asr.core.cli import params as paramsfunc
    from asr.core import read_json
    from pathlib import Path
    runner = CliRunner()
    result = runner.invoke(paramsfunc, ['asr.c2db.relax:main', 'd3=True'])
    assert result.exit_code == 0
    p = Path('params.json')
    assert p.is_file()
    params = read_json('params.json')
    assert params['asr.c2db.relax:main']['d3'] is True

    result = runner.invoke(
        paramsfunc,
        ['asr.c2db.gs:calculate', 'calculator={"name": "testname", ...}']
    )
    assert result.exit_code == 0
    params = read_json('params.json')
    assert params['asr.c2db.relax:main']['d3'] is True
    assert params['asr.c2db.gs:calculate']['calculator']['name'] == 'testname'
    assert params['asr.c2db.gs:calculate']['calculator']['charge'] == 0

    result = runner.invoke(paramsfunc, ['asr.c2db.relax:main', 'd3=False'])
    assert result.exit_code == 0
    params = read_json('params.json')
    assert params['asr.c2db.relax:main']['d3'] is False
    assert params['asr.c2db.gs:calculate']['calculator']['name'] == 'testname'
    assert params['asr.c2db.gs:calculate']['calculator']['charge'] == 0


@pytest.mark.xfail
def test_asterisk_syntax():
    from asr.core.cli import params as paramsfunc
    from asr.core import read_json
    runner = CliRunner()
    result = runner.invoke(paramsfunc, ['*:kptdensity', '12'])
    assert result.exit_code == 0
    params = read_json('params.json')
    assert params["asr.polarizability"]["kptdensity"] == 12
    for value in params.values():
        if 'kptdensity' in value:
            assert value['kptdensity'] == 12


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
def test_setup_params_recurse_dict(asr_tmpdir):
    from asr.core.cli import params as paramsfunc
    from asr.core import read_json

    runner = CliRunner()
    result = runner.invoke(
        paramsfunc,
        ['asr.c2db.gs:calculate',
         'calculator={"name": "testname", "mode": {"ecut": 400, ...}, ...}']
    )
    assert result.exit_code == 0
    params = read_json('params.json')
    assert params['asr.c2db.gs:calculate']['calculator']['name'] == 'testname'
    assert params['asr.c2db.gs:calculate']['calculator']['mode']['name'] == 'pw'
    assert params['asr.c2db.gs:calculate']['calculator']['mode']['ecut'] == 400
