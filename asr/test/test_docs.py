import os
import pytest
import pathlib

from asr.core import chdir
from .utils import (
    run_shell_command, get_commands_and_outputs,
    get_asr_home_path, get_asr_library_path
)


@pytest.fixture
def command_outputs(request):
    path = request.param
    txt = path.read_text()
    lines = txt.split('\n')
    return get_commands_and_outputs(lines)


directory = pathlib.Path('docs/src')
tutorials = []
rstfiles = list(directory.rglob('tutorials/getting-started.rst'))


@pytest.mark.parametrize("command_outputs", rstfiles, indirect=True)
def test_tutorial(command_outputs, tmpdir):
    my_env = os.environ.copy()
    asrhome = get_asr_home_path()
    my_env['ASRHOME'] = asrhome
    print('ASRHOME', asrhome)
    with chdir(tmpdir):
        asrlib = get_asr_library_path()
        my_env['ASRLIB'] = asrlib
        print(f'Running in {tmpdir}')
        for _, _, command, output in command_outputs:
            print(command)
            actual_output = run_shell_command(command, env=my_env)
            # Below is a hack for removing printed uids and other stuff
            # that change on every run. A better solution can probably
            # be found.
            new_output = prepare_output_for_comparison(output)
            new_actual_output = prepare_output_for_comparison(actual_output)
            assert new_output == new_actual_output, (output, actual_output)


def prepare_output_for_comparison(output):
    new_output = []
    for il, line in enumerate(output):
        line, *_ = line.split('uid')
        line, *_ = line.split('execution_')
        line, *_ = line.split('created')
        line, *_ = line.split('modified')
        line, *_ = line.split('latest_revision')
        line, *_ = line.split('version=')
        new_output.append(line)
    return new_output
