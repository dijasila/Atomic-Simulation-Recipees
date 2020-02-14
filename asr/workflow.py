from asr.core import command


@command()
def main():
    from asr.core import get_recipes

    recipes = get_recipes()
    recipes = filter(lambda x: (x.name.count('.') == 1
                                and '@' not in x.name),
                     recipes)
    for recipe in recipes:
        if recipe.name == 'asr.workflow':
            continue
        if not recipe.done:
            recipe()


if __name__ == '__main__':
    main.cli()
