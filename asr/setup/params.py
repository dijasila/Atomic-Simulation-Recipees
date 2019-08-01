import click
from asr.utils import command, argument


@command('asr.setup.params',
         save_results_file=False)
@argument('params', nargs=-1,
          metavar='recipe:option arg recipe:option arg')
def main(params):
    """Compile a params.json file with all options and defaults.

    This recipe compiles a list of all options and their default
    values for all recipes to be used for manually changing values
    for specific options."""
    import json
    from pathlib import Path
    from asr.utils import get_recipes, ASRCommand

    p = Path('params.json')
    assert not p.exists(), 'params.json already exists!'

    defparamdict = {}
    
    recipes = get_recipes(sort=True)
    for recipe in recipes:
        if not recipe.main:
            continue
        defparams = {}
        ctx = click.Context(ASRCommand)
        opts = recipe.main.get_params(ctx)
        for opt in opts:
            if opt.name == 'help':
                continue
            defparams[opt.name] = opt.get_default(ctx)

        defparamdict[recipe.name] = defparams

    paramdict = {}
    if params:
        # Find recipe:option
        options = params[::2]
        args = params[1::2]

        for option, value in zip(options, args):
            assert ':' in option, 'You have to use the recipe:option syntax'
            recipe, option = option.split(':')
            assert option, 'You have to provide an option'
            assert recipe, 'You have to provide a recipe'

            assert recipe in defparamdict, \
                f'This is an unknown recipe: {recipe}'
            assert option in defparamdict[recipe], \
                f'This is an unknown option: {recipe}:{option}'

            if recipe not in paramdict:
                paramdict[recipe] = {}

            assert option not in paramdict[recipe], \
                'You cannot write the same option twice'
            paramdict[recipe][option] = \
                type(defparamdict[recipe][option])(value)

    if not paramdict:
        paramdict = defparamdict
    p.write_text(json.dumps(paramdict, indent=4))
    return paramdict


tests = [
    {'cli': ['asr run setup.params']},
    {'cli': ['asr run setup.params asr.relax:ecut 300'],
     'results': [{'file': 'params.json',
                  'asr.relax:ecut': (250, 0.1)}], 'fails': True},
    {'cli': ['asr run setup.params :ecut 300'], 'fails': True},
    {'cli': ['asr run setup.params asr.relax: 300'], 'fails': True},
    {'cli': ['asr run setup.params asr.relax:ecut asr.gs:ecut 300'],
     'fails': True},
]

group = 'setup'


if __name__ == '__main__':
    main()
