import os
import tempfile
import pathlib
import textwrap
from asr.core import chdir
from asr.test.utils import (
    run_shell_command, get_command_and_output_ranges,
    get_asr_home_path, get_asr_library_path,
)

def update_tutorials():
    tutorials = ['src/tutorials/getting-started.rst']
    for tutorial in tutorials:
        path = pathlib.Path(tutorial).absolute()
        env = os.environ.copy()
        asrhome = get_asr_home_path()
        dirpath = tempfile.mkdtemp()
        with chdir(dirpath):
            asrlib = get_asr_library_path()
            env['ASRHOME'] = asrhome
            env['ASRLIB'] = asrlib
            txt = path.read_text()
            lines = txt.split('\n')
            ranges = get_command_and_output_ranges(lines)
            outputs = []
            for ic, io in ranges:
                command = lines[ic][5:]
                output = run_shell_command(command, env=env)
                output = ['   ' + tmp for tmp in output]
                outputs.append(output)

            for (ic, io), output in (
                list(zip(ranges, outputs))[::-1]
            ):
                lines = lines[:ic + 1] + output + lines[io:]
            new_text = '\n'.join(lines)
            path.write_text(new_text)
            print(new_text)


if __name__ == '__main__':
    update_tutorials()

