def get_recipes():
    import importlib
    from pathlib import Path

    files = Path(__file__).parent.glob('*.py')
    exclude = ['__init__.py', '__main__.py', 'utils.py']
    recipes = []
    for file in files:
        is_recipe = True
        if str(file.name) in exclude:
            is_recipe = False

        if is_recipe:
            name = file.with_suffix('').name
            module = importlib.import_module(f'asr.{name}')
            recipes.append(module)
    return recipes


def summary(content, indent=0, title=None, pad=2):
    colwidth_c = []
    for row in content:
        if isinstance(row, str):
            continue
        for c, element in enumerate(row):
            nchar = len(element)
            try:
                colwidth_c[c] = max(colwidth_c[c], nchar)
            except IndexError:
                colwidth_c.append(nchar)

    output = ''
    if title:
        output = f'\n{title}\n'
    for row in content:
        out = ' ' * indent
        if isinstance(row, str):
            output += f'\n{row}\n'
            continue
        for colw, desc in zip(colwidth_c, row):
            out += f'{desc: <{colw}}' + ' ' * pad
        output += out
        output += '\n'

    return output


def check_recipes():
    print('Checking recipes...')
    recipes = get_recipes()

    attributes = ['main',
                  'creates',
                  'collect_data',
                  'webpanel',
                  'resources']

    groups = ['Structure', 'Property',
              'Postprocessing', 'Utility']
    panel = []
    panel.append(['name', *attributes])
    for group in groups:
        panel.append(f'{group} recipes')
        for recipe in recipes:
            if not recipe.group == group:
                continue
            status = [recipe.__name__]
            for attr in attributes:
                if hasattr(recipe, attr):
                    status.append('.')
                else:
                    status.append('N')
            panel.append(status)

    pretty_output = summary(panel)
    print(pretty_output)


if __name__ == '__main__':
    check_recipes()
