from asr.core import command, argument


tests = [
    {'cli': ['asr run setup.params']},
    {'cli': ['asr run "setup.params asr.relax:ecut 300"'],
     'results': [{'file': 'params.json',
                  'asr.relax:ecut': (250, 0.1)}], 'fails': True},
    {'cli': ['asr run "setup.params :ecut 300"'], 'fails': True},
    {'cli': ['asr run "setup.params asr.relax: 300"'], 'fails': True},
    {'cli': ['asr run "setup.params asr.relax:ecut asr.gs:ecut 300"'],
     'fails': True},
]


@command('asr.setup.params',
         tests=tests)
@argument('params', nargs=-1,
          metavar='recipe:option arg recipe:option arg')
def main(params=None):
    """Compile a params.json file with all options and defaults.

    This recipe compiles a list of all options and their default
    values for all recipes to be used for manually changing values
    for specific options."""
    import json
    from pathlib import Path
    from asr.utils import get_recipes, read_json

    defparamdict = {}
    
    recipes = get_recipes()
    for recipe in recipes:
        defparams = recipe.defparams
        defparamdict[recipe.name] = defparams

    p = Path('params.json')
    if p.is_file():
        paramdict = read_json('params.json')
    else:
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

            paramdict[recipe][option] = \
                type(defparamdict[recipe][option])(value)

    if not paramdict:
        paramdict = defparamdict
    p.write_text(json.dumps(paramdict, indent=4))
    return paramdict


if __name__ == '__main__':
    main.cli()
