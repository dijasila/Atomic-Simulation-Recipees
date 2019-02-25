import importlib
from pathlib import Path
from argparse import ArgumentParser
from ase.cli.main import Formatter

parser = ArgumentParser(prog='mcr',
                        description='Material characterization recipies',
                        formatter_class=Formatter)

subparsers = parser.add_subparsers(title='Sub-commands', dest='command')

pathlist = Path('.').glob('*.py')
for path in pathlist:
    name = str(path)[:-3]
    module = importlib.import_module('.' + name, package='mcs')
    try:
        sp = module.get_parser()
    except AttributeError:
        continue
    subparsers.add_parser(name, parents=[sp], add_help=False,
                          help=sp.description)
parser.parse_args()
