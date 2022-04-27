import pytest

from click.testing import CliRunner
from asr.core.cli import cli


@pytest.mark.ci
def test_asr():
    """Test the main CLI."""
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0
    assert 'Usage: cli [OPTIONS] COMMAND [ARGS]...' in result.output
    help_result = runner.invoke(cli, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
    help_result = runner.invoke(cli, ['-h'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


@pytest.mark.xfail(reason='CLI port')
@pytest.mark.ci
def test_asr_run(asr_tmpdir_w_params):
    import pathlib

    runner = CliRunner()
    result = runner.invoke(cli, ['run', '-h'])
    assert result.exit_code == 0
    assert 'Usage: cli run [OPTIONS] COMMAND [FOLDERS]' in result.output

    help_result = runner.invoke(cli, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output

    help_result = runner.invoke(cli, ['-h'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output

    result = runner.invoke(cli, ['run', '--dry-run', 'structureinfo'])
    assert result.exit_code == 0
    assert 'Would run asr.structureinfo:main in 1 folders.' in result.output

    pathlib.Path("folder1").mkdir()
    pathlib.Path("folder2").mkdir()
    result = runner.invoke(cli, ['run', '--dry-run',
                                 'structureinfo',
                                 'folder1', 'folder2'])

    assert ('Number of folders: 2\nWould run asr.structureinfo:main'
            ' in 2 folders.\n') in result.output

    assert result.exit_code == 0

    result = runner.invoke(cli, ['run', '--dry-run', '--njobs', '2',
                                 'structureinfo',
                                 'folder3', 'folder4'])
    assert result.exit_code == 0
    assert 'Number of folders: 2\nNumber of jobs: 2\n' in result.output


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
def test_asr_list():
    runner = CliRunner()
    result = runner.invoke(cli, ['list'])
    assert result.exit_code == 0
    assert 'Name' in result.output
    assert 'Description' in result.output


@pytest.mark.xfail(reason='CLI port')
@pytest.mark.ci
def test_asr_results_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['results', '-h'])
    assert result.exit_code == 0
    assert 'Usage: cli results [OPTIONS] [SELECTION]' in result.output


@pytest.mark.xfail
@pytest.mark.ci
def test_asr_results_bandstructure(asr_tmpdir, mockgpaw, mocker):
    from asr.c2db.gs import main as calculate_gs
    from .materials import BN
    import matplotlib.pyplot as plt
    import gpaw
    mocker.patch.object(gpaw.GPAW, "_get_band_gap")
    mocker.patch.object(gpaw.GPAW, "_get_fermi_level")
    gpaw.GPAW._get_fermi_level.return_value = 0.5
    gpaw.GPAW._get_band_gap.return_value = 1

    mocker.patch.object(plt, 'show')
    BN.write('structure.json')
    calculate_gs()
    runner = CliRunner()
    result = runner.invoke(cli, ['results', 'asr.c2db.gs'])
    assert result.exit_code == 0, result
    assert 'Saved figures: bz-with-gaps.png' in result.output


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
def test_asr_cache_ls(asr_tmpdir_w_params):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['cache', 'ls'])

    assert result.exit_code == 0
    assert result.output == 'name parameters result\n'


@pytest.fixture()
def cache_with_record(fscache, record):
    fscache.add(record)
    return record, fscache


@pytest.mark.xfail(reason='not now')
@pytest.mark.ci
@pytest.mark.parametrize(
    'args,output,final_record_count',
    [
        ([], 'Deleted 1 record(s)', 0),
        (['--dry-run'], 'Would delete 1 record(s).', 1),
        (['-z'], 'Would delete 1 record(s).', 1),
    ])
def test_asr_cache_rm(
        cache_with_record,
        args, output, final_record_count,
):
    record, cache = cache_with_record
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['cache', 'rm', f'uid={record.uid}'] + args)

    assert result.exit_code == 0
    assert output in result.output

    assert len(cache.select()) == final_record_count
