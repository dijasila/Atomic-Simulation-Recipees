import click
from asr.utils import command, argument


@command('asr.setup.scanparams',
         save_results_file=False)
@argument('scanparams', nargs=-1,
          metavar='recipe:option arg arg arg recipe:option arg arg arg')
def main(scanparams):
    """Make new folders different parameters (for example for convergence test)

    This function will take a number of arguments in the syntax
    recipe:option arg arg arg and setup a number of folders named
    "scanparams*/" (where * is an integer) with a params.json file.

    If you set multiple options setup.scanparams will test all combinations
    of parameters.

    The function checks if any of the parameter combinations you ask for
    already exists and skips these parameter combinations if this is the case.

    Examples:
    \b
    Test different kpoint density in the relax recipe
        asr run setup.scanparams asr.relax:kptdensity 3 4 5
    Test combination of kpoint densities and planewave cutoff in relax:
        asr run setup.scanparams asr.relax:kptdensity 3 4 5 \
            asr.relax:ecut 300 400 500
    """
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
                print(f'{params} already exists in {p}')
                allparams.pop(i)
                break
        maxind = max(maxind, int(str(p)[10:]) + 1)

    if not allparams:
        print('All parameter combinations already exists. '
              'Generated no new folders.')

    for j, params in enumerate(allparams):
        folder = Path(f'scanparams{maxind + j}')
        print(f'Generating {folder} with params: {params}')
        folder.mkdir()
        write_json(str(folder / 'params.json'), params)


group = 'setup'


if __name__ == '__main__':
    main()
