import click
from asr.utils import command, argument, option


@command('asr.setup.scanparams',
         save_results_file=False)
@argument('scanparams', nargs=-1,
          metavar='recipe:option arg arg arg and recipe:option arg arg arg')
@option('--separate', is_flag=True, help='Vary parameters separate')
def main(scanparams, separate):
    """Make a new params file"""
    from pathlib import Path
    from asr.utils import get_recipes, ASRCommand

    paramdict = {}
    recipes = get_recipes(sort=True)
    for recipe in recipes:
        if not recipe.main:
            continue
        params = {}
        ctx = click.Context(ASRCommand)
        opts = recipe.main.get_params(ctx)
        for opt in opts:
            if opt.name == 'help':
                continue
            params[opt.name] = opt.get_default(ctx)

        paramdict[recipe.name] = params

    # Find asr.recipe:option
    optioninds = []
    for i, arg in enumerate(scanparams):
        if isinstance(arg, str):
            if arg.startswith('asr.'):
                assert ':' in arg, 'You have to use the recipe:option syntax'
                optioninds.append(i)
    optioninds.append(len(scanparams))

    splitscanparams = []
    scanparamsdict = {}
    for start, end in zip(optioninds[:-1], optioninds[1:]):
        params = scanparams[start:end]
        values = params[1:]
        assert ':' in params[0], 'You have to use the recipe:option syntax'
        assert len(values), 'You have to provide at least one parameter'
        recipe, option = params[0].split(':')
        assert recipe in paramdict, f'Unknown recipe: {recipe}'
        assert option in paramdict[recipe], \
            f'Unknown option: {recipe}:{option}'

        if recipe not in scanparamsdict:
            scanparamsdict[recipe] = {}
        assert option not in scanparamsdict[recipe], \
            'You cannot set {recipe}:{option} twice'

        newparams = [params[0]]
        for value in values:
            newparams.append(type(paramdict[recipe][option])(value))
        scanparamsdict[recipe][option] = newparams[1:]
        splitscanparams += [newparams]

    from itertools import product
    parameternames = [params[0] for params in splitscanparams]
    scanparams = [params[1:] for params in splitscanparams]
    allparams = []
    for values in product(*scanparams):
        params = {}
        for name, value in zip(parameternames, values):
            recipe, option = name.split(':')
            if recipe not in params:
                params[recipe] = {}
            assert option not in params[recipe], 'This should not happen!'
            params[recipe][option] = value
        allparams.append(params)

    for i, params in enumerate(allparams):
        folder = f'scanparams{i}'
        print(f'Generated {folder}/params.json with {params}')

    # Find parameter combinations that have already been used
    from asr.utils import read_json, write_json
    newparams = []
    maxind = 0
    for p in Path().glob('scanparams*'):
        if not p.is_dir():
            continue
        assert (p / 'params.json').is_file(), \
            "{p}/params.json should exist but it doesn't"
        params = read_json(str(p / 'params.json'))

        for i, generatedparams in enumerate(allparams):
            if params == generatedparams:
                allparams.pop(i)
                break
        maxind = max(maxind, str(p)[len('scanparams'):])

    for params in allparams:
        folder = Path(f'scanparams{i}')
        folder.mkdir()
        write_json(str(folder / 'params.json'), params)


group = 'setup'


if __name__ == '__main__':
    main()
