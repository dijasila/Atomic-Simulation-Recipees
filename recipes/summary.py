def summary(**kwargs):
    from pathlib import Path
    from importlib import import_module
    
    pathlist = Path(__file__).parent.glob('*.py')
    for path in pathlist:
        name = path.with_suffix('').name
        module = import_module('.recipies.' + name, package='mcr')

        try:
            module.print_results()
        except AttributeError:
            continue


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Print summary of results')

    return parser


def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
    summary(**vars(args))
