import os
import pytest
import pathlib
import subprocess

from asr.core import chdir


@pytest.fixture
def command_outputs(request):
    import textwrap
    path = request.param
    txt = path.read_text()
    lines = txt.split('\n')
    command_lines = []
    for il, line in enumerate(lines):
        if line.startswith('   $ '):
            command_lines.append(il)

    commands_outputs = []
    for il in command_lines:
        output = []
        for line in lines[il + 1:]:
            if line.startswith('   ') and not line.startswith('   $ '):
                output.append(line)
            else:
                break
        command = lines[il][5:]
        if output:
            output = textwrap.dedent('\n'.join(output)).split('\n')
        commands_outputs.append((command, output))
    return commands_outputs


directory = pathlib.Path('docs/src')
tutorials = []
rstfiles = list(directory.rglob('tutorials/getting-started.rst'))


@pytest.mark.parametrize("command_outputs", rstfiles, indirect=True)
def test_tutorial(command_outputs, tmpdir):
    import asr
    my_env = os.environ.copy()
    asrhome = pathlib.Path(asr.__file__).parent.parent
    my_env['ASRHOME'] = asrhome
    print('ASRHOME', asrhome)
    with chdir(tmpdir):
        completed_process = subprocess.run(
            'python3 -c "import asr; print(asr.__file__)"',
            capture_output=True, shell=True)
        asrlib = pathlib.Path(completed_process.stdout.decode()).parent
        my_env['ASRLIB'] = asrlib
        print(f'Running in {tmpdir}')
        for command, output in command_outputs:
            print(command)
            completed_process = subprocess.run(
                command, capture_output=True, env=my_env,
                shell=True)
            try:
                actual_output = completed_process.stdout.decode()
                assert not completed_process.returncode
            except UnicodeDecodeError:
                actual_output = completed_process.stderr.decode()
            actual_output = actual_output.split('\n')
            if actual_output[-1] == '':
                actual_output.pop()
            # This is a hack for removing printed uids since they change
            # on every run. A better solution can probably be found.
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
        new_output.append(line)
    return new_output
