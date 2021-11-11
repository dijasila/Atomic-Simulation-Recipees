"""Generate parameter file 'params.json'."""
from typing import Union
from asr.core import command, argument, ASRResult


@command('asr.setup.params',
         save_results_file=True)
@argument('params', nargs=-1, type=str,
          metavar='recipe:option arg recipe:option arg')
def main(params: Union[str, None] = None) -> ASRResult:
    """Compile a params.json file with all options and defaults.

    This recipe compiles a list of all options and their default
    values for all recipes to be used for manually changing values
    for specific options.
    """
    from pathlib import Path
    from asr.core import get_recipes, read_json, write_json
    from ast import literal_eval
    from fnmatch import fnmatch
    import copy
    from asr.core import recursive_update

    defparamdict = {}
    recipes = get_recipes()
    for recipe in recipes:
        defparams = recipe.get_defaults()
        defparamdict[recipe.name] = defparams

    p = Path('params.json')
    if p.is_file():
        paramdict = read_json('params.json')
        # The existing valus in paramdict set the new defaults
        recursive_update(defparamdict, paramdict)
    else:
        paramdict = {}

    if isinstance(params, (list, tuple)):
        # Find recipe:option
        tmpoptions = params[::2]
        print(tmpoptions)
        tmpargs = params[1::2]
        #assert len(tmpoptions) == len(tmpargs), \
        #    'You must provide a value for each option'
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

            if recipe not in paramdict:
                paramdict[recipe] = {}

            paramtype = type(defparamdict[recipe][option])
            if paramtype == dict:
                value = value.replace('...', 'None:None')
                val = literal_eval(value)
            elif paramtype == bool:
                val = literal_eval(value)
            else:
                val = paramtype(value)
            paramdict[recipe][option] = val
    elif isinstance(params, dict):
        paramdict.update(copy.deepcopy(params))
    else:
        raise NotImplementedError(
            'asr.setup.params is only compatible with'
            f'input lists and dict. Input params: {params}'
        )

    for recipe, options in paramdict.items():
        assert recipe in defparamdict, \
            f'This is an unknown recipe: {recipe}'

        for option, value in options.items():
            assert option in defparamdict[recipe], \
                f'This is an unknown option: {recipe}:{option}'
            if isinstance(value, dict):
                recursive_update(value, defparamdict[recipe][option])
                paramdict[recipe][option] = value

    if paramdict:
        write_json(p, paramdict)
    return paramdict


if __name__ == '__main__':
    main.cli()
