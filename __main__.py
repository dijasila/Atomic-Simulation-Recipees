import importlib.util
import json
from pathlib import Path
from argparse import (ArgumentParser,
                      ArgumentDefaultsHelpFormatter,
                      RawTextHelpFormatter)
from .utilities import (get_folder_file,
                        get_home_folders,
                        write_home_folders)
from .recipe import Recipe


# Create ~/.myrecipes/ if it's not there:
f = Path.home() / '.myrecipes'
if not f.is_dir():
    f.mkdir()


# Create ~/.myrecipes/folders.txt if it's not there:
folderfile = get_folder_file()
if not folderfile.is_file():
    folders = get_home_folders()
    write_home_folders(folders)

prog = 'myrecipes'
usage = f'{prog} [-h] collection recipe ...'

parser = ArgumentParser(prog=prog,
                        usage=usage,
                        description='Recipes for Materials Research',
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('collection', type=str,
                    help='Collection to find recipe is located')

subparsers = parser.add_subparsers(dest='command', metavar='recipe',
                                   help='Recipe to be cooked')

infos = {}
collections = {}
for folder in get_home_folders():
    recipes = {}
    info = Path(folder) / 'info.json'
    if not info.is_file():
        continue
    
    dct = json.load(open(str(info), 'r'))
    alias = dct['alias']
    assert alias not in collections, print('This alias already exists!')
    infos[alias] = dct
    collections[alias] = recipes
    pathlist = folder.glob('*.py')
    for path in pathlist:
        name = path.with_suffix('').name
        spec = importlib.util.spec_from_file_location('', str(path))
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except FileNotFoundError:
            continue

        try:
            sp = module.parser
            main = module.main
        except AttributeError:
            continue
        sp = subparsers.add_parser(name, parents=[sp], add_help=False)
        sp.formatter_class = ArgumentDefaultsHelpFormatter

        recipes[name] = Recipe(module)


# Find external recipies
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


# Sort recipes and compile summary
epilog = """ \n
Available Recipies:
-------------------

Called as:
>>> myrecipes <alias> <recipe> [args]

For example:
>>> myrecipes core add folder
>>> myrecipes core remove folder

"""
for key in collections:
    info = infos[key]
    epilog += f'{info["title"]} (alias: {info["alias"]})\n'
    recipes = collections[key]
    sorted_recipes = [recipes[key] for key in recipes]
    sorted_recipes.sort(key=lambda x: x.group)

    group = ''
    content = []
    for r in sorted_recipes:
        p = r.parser
        if r.group != group:
            group = r.group
            if group in info['group_descriptions']:
                desc = f'{group} ({info["group_descriptions"][group]})'
            else:
                desc = group
            content.append(desc)
        desc = [r.name, r.short_description]
        if r.dependencies:
            desc[-1] += f' {r.dependencies}'
        content.append(desc)
    epilog += summary(content, indent=4) + '\n'
parser.epilog = epilog

args = parser.parse_args()

if args.command:
    # Then execute recipe
    recipes = collections[args.collection]
    module = recipes[args.command]
    sp = module.parser
    args = vars(args)
    # Pop the arguments that the subparsers dont know
    args.pop('command')
    args.pop('collection')
    f = module.main
    f(args)
elif args.command is None:
    parser.print_usage()
