import argparse


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


short_description = 'Print summary of results'
parser = argparse.ArgumentParser(description=short_description)


def main(args):
    summary(**vars(args))


if __name__ == '__main__':
    args = vars(parser.parser_args())
    main(args)
