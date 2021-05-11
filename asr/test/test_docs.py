import pytest
import pathlib
import subprocess
import shlex

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
        if not output:
            output = ['']
        output = textwrap.dedent('\n'.join(output)).split('\n')
        commands_outputs.append((lines[il][5:], output))
    return commands_outputs


directory = pathlib.Path('docs/src')
tutorials = []
rstfiles = list(directory.rglob('tutorials/getting-started.rst'))


@pytest.mark.parametrize("command_outputs", rstfiles, indirect=True)
def test_tutorial(command_outputs, tmpdir):
    with chdir(tmpdir):
        print(f'Running in {tmpdir}')
        for command, output in command_outputs:
            print(command)
            completed_process = subprocess.run(
                shlex.split(command), capture_output=True)
            try:
                actual_output = completed_process.stdout.decode()
            except UnicodeDecodeError:
                actual_output = completed_process.stderr.decode()
            actual_output = actual_output.split('\n')
            assert output == actual_output, (output, actual_output)
