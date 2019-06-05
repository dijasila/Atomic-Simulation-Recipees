import click
from asr.utils import command


@command('asr.setup.params',
         save_results_file=False)
def main():
    """Compile a params.json file with all options and defaults.

    This recipe compiles a list of all options and their default
    values for all recipes to be used for manually changing values
    for specific options."""
    import json
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

    p = Path('params.json')
    assert not p.exists(), 'params.json already exists!'
    p.write_text(json.dumps(paramdict, indent=4))


group = 'setup'


if __name__ == '__main__':
    main()
