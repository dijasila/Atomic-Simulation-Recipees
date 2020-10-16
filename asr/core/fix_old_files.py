import os
import pathlib
import click
from asr.core import read_json, ASRResult


def is_results_file(path: pathlib.Path):
    name = path.name
    return name.startswith('results-') and name.endswith('.json')


@click.command()
@click.option('--fixup', is_flag=True, help='Fix bad files.')
def find_bad_results_files(fixup=False):
    badresultfiles = []
    for root, dirs, files in os.walk("."):
        for name in files:
            path = pathlib.Path(root) / name
            abspath = path.absolute()
            print(abspath, flush=True)
            if is_results_file(path):
                content = read_json(path)
                assert isinstance(content, (dict, ASRResult)), (type(content), content)
                if '__asr_name__' not in content:
                    badresultfiles.append(abspath)
                    print('IS BAD', flush=True)


if __name__ == '__main__':
    find_bad_results_files()
