import os
import pathlib
import click
from asr.core import read_json


def is_results_file(path: pathlib.Path):
    name = path.name
    return name.startswith('results-') and name.endswith('.json')


@click.command()
@click.option('--fixup', help='Fix bad files.')
def find_bad_results_files():
    badresultfiles = []
    for root, dirs, files in os.walk("."):
        for name in files:
            path = pathlib.Path(root) / name
            if is_results_file(path):
                content = read_json(path)
                assert isinstance(content, dict), f'Not a dict={content}'
                if '__asr_name__' not in content:
                    badresultfiles.append(path)

    for path in badresultfiles:
        print(path.absolute())


if __name__ == '__main__':
    find_bad_results_files()
