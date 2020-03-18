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
    for specific options.
    """
    import json
    from pathlib import Path
    from asr.core import get_recipes, read_json, parse_dict_string
    from ast import literal_eval
    from fnmatch import fnmatch
    import copy

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
        tmpoptions = params[::2]
        tmpargs = params[1::2]
        assert len(tmpoptions) == len(tmpargs), \
            'You must provide a value for each option'
        options = []
        args = []
        for tmpoption, tmparg in zip(tmpoptions, tmpargs):
            assert ':' in tmpoption, 'You have to use the recipe:option syntax'
            recipe, option = tmpoption.split(':')
            if '*' in recipe:
                for tmprecipe in defparamdict:
                    if not fnmatch(tmprecipe, recipe):
                        continue
                    if option in defparamdict[tmprecipe]:
                        options.append(f'{tmprecipe}:{option}')
                        args.append(tmparg)
            else:
                options.append(tmpoption)
                args.append(tmparg)

        for option, value in zip(options, args):
            recipe, option = option.split(':')

            assert option, 'You have to provide an option'
            assert recipe, 'You have to provide a recipe'

            assert recipe in defparamdict, \
                f'This is an unknown recipe: {recipe}'
            assert option in defparamdict[recipe], \
                f'This is an unknown option: {recipe}:{option}'

            if recipe not in paramdict:
                paramdict[recipe] = {}

            paramtype = type(defparamdict[recipe][option])
            if paramtype == dict:
                dct = copy.deepcopy(defparamdict[recipe][option])
                val = parse_dict_string(value, dct=dct)
            elif paramtype == bool:
                val = literal_eval(value)
            else:
                val = paramtype(value)
            paramdict[recipe][option] = val

    if paramdict:
        p.write_text(json.dumps(paramdict, indent=1))
    return paramdict


if __name__ == '__main__':
    main.cli()
