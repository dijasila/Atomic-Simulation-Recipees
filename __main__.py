import importlib
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(prog='mcr',
                        description='Material characterization recipies',
                        formatter_class=ArgumentDefaultsHelpFormatter)

subparsers = parser.add_subparsers(title='Recipies', dest='command')

pathlist = Path(__file__).parent.glob('./recipies/*.py')
functions = {}
functionparsers = {}

for path in pathlist:
    name = path.with_suffix('').name
    module = importlib.import_module('.recipies.' + name, package='mcr')
    try:
        sp = module.get_parser()
        main = module.main
    except AttributeError:
        continue
    sp = subparsers.add_parser(name, parents=[sp], add_help=False,
                               help=sp.description)
    sp.formatter_class = ArgumentDefaultsHelpFormatter
    functions[name] = main
    functionparsers[name] = sp
    
args = parser.parse_args()
if args.command:
    f = functions[args.command]
    sp = functionparsers[name]
    knownargs = sp.parse_known_args()[0]
    f(vars(knownargs))
