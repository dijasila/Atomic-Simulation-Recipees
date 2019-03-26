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
            output += f'\n  {row}\n'
            continue
        for colw, desc in zip(colwidth_c, row):
            out += f'{desc: <{colw}}' + ' ' * pad
        output += out
        output += '\n'

    return output


def check_recipes():
    print('Checking recipes...')
    from asr import recipes

    attributes = ['main',
                  'collect_data',
                  'webpanel',
                  'resources']

    panel = []
    panel.append(['name', *attributes])
    for recipe in recipes:
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
