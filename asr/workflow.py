from asr.core import command


@command('asr.workflow')
def main():
    from asr.core import get_recipes

    recipes = get_recipes()
    recipes = filter(lambda x: ('database' not in x.name
                                and '@' not in x.name
                                and x.name != 'asr.workflow'),
                     recipes)

    while any(not recipe.done for recipe in recipes):
        for recipe in recipes:
            try:
                if recipe.is_requirements_met() and not recipe.done:
                    recipe()
            except Exception as e:
                print(e)


if __name__ == '__main__':
    main.cli()
