def main():
    """Print summary of results"""
    from pathlib import Path
    from importlib import import_module
    
    pathlist = Path(__file__).parent.glob('*.py')
    for path in pathlist:
        name = path.with_suffix('').name
        module = import_module('asr.' + name)

        try:
            module.print_results()
        except AttributeError:
            continue


group = 'Postprocessing'

if __name__ == '__main__':
    main()
