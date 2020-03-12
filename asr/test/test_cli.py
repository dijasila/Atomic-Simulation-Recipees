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


@pytest.mark.ci
def test_asr_run(separate_folder):
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

    result = runner.invoke(cli, ['run', '--dry-run', 'setup.params'])
    assert result.exit_code == 0
    assert 'Would run asr.setup.params@main in 1 folders.' in result.output

    pathlib.Path("folder1").mkdir()
    pathlib.Path("folder2").mkdir()

    result = runner.invoke(cli, ['run',
                                 'setup.params asr.relax:d3 True',
                                 'folder1', 'folder2'])
    assert result.exit_code == 0
    assert pathlib.Path("folder1", "params.json").is_file()
    assert pathlib.Path("folder2", "params.json").is_file()

    pathlib.Path('str1.json').write_text("")
    result = runner.invoke(cli, ['run', '--shell', 'mv str1.json str2.json'])
    assert pathlib.Path("str2.json").is_file()

    pathlib.Path("folder3").mkdir()
    pathlib.Path("folder4").mkdir()

    result = runner.invoke(cli, ['run', '--jobs', '2',
                                 'setup.params asr.relax:d3 True',
                                 'folder3', 'folder4'])
    assert result.exit_code == 0
    assert pathlib.Path("folder3", "params.json").is_file(), result
    assert pathlib.Path("folder4", "params.json").is_file(), result


@pytest.mark.ci
def test_asr_list():
    runner = CliRunner()
    result = runner.invoke(cli, ['list'])
    assert result.exit_code == 0
    assert 'Name' in result.output
    assert 'Description' in result.output


@pytest.mark.ci
def test_asr_results(separate_folder, mockgpaw):
    runner = CliRunner()
    result = runner.invoke(cli, ['results', '-h'])
    assert result.exit_code == 0
    assert 'Usage: asr results [OPTIONS] NAME' in result.output
