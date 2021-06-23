import pathlib
import os
import subprocess


def get_asr_library_path():
    completed_process = subprocess.run(
        'python3 -c "import asr; print(asr.__file__)"',
        capture_output=True, shell=True)
    asrlib = pathlib.Path(completed_process.stdout.decode()).parent
    return asrlib


def get_asr_home_path():
    import asr
    asrhome = pathlib.Path(asr.__file__).parent.parent
    return asrhome


def run_shell_command(command, env=None):
    if env is None:
        env = os.environ.copy()
    completed_process = subprocess.run(
        command, capture_output=True, env=env,
        shell=True,
    )
    try:
        output = completed_process.stdout.decode()
        assert not completed_process.returncode
    except UnicodeDecodeError:
        output = completed_process.stderr.decode()
    output = output.split('\n')
    if output[-1] == '':
        output.pop()
    return output


def get_command_and_output_ranges(lines):
    command_lines = []
    for il, line in enumerate(lines):
        if line.startswith('   $ '):
            command_lines.append(il)

    ranges = []
    for il in command_lines:
        output = []
        for io, line in enumerate(lines[il + 1:], start=il + 1):
            if line.startswith('   ') and not line.startswith('   $ '):
                output.append(line)
            else:
                rng = (il, io)
                break
        ranges.append(rng)
    return ranges


def get_commands_and_outputs(lines):
    command_lines = []
    for il, line in enumerate(lines):
        if line.startswith('   $ '):
            command_lines.append(il)

    commands_outputs = []
    ranges = get_command_and_output_ranges(lines)
    for ic, io in ranges:
        command = lines[ic][5:]
        output = [line[3:] for line in lines[ic + 1:io]]
        commands_outputs.append((ic, io, command, output))
    return commands_outputs
