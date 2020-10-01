"""Autogenerate documentation for recipes and modules."""
from pathlib import Path
import importlib
import inspect
import os


def get_modules_from_path(path: str, recursive=False):
    """Get modules from path."""
    if recursive:
        modules = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                p = Path(root) / filename
                if p.suffix == '.py':
                    modules.append(p)
        return modules
    else:
        return list(Path(path).glob('*.py'))


def get_names_from_paths(paths):
    """Get module names from path."""
    return [str(path.with_suffix('')).replace('/', '.')
            for path in paths]


def make_section(title, names, link):
    """Make a recipe section."""
    return ([f'.. _{link}:',
             '',
             title,
             '-' * len(title),
             '',
             '.. autosummary::',
             '   :toctree: .',
             '']
            + [f'   {name}'
               for name in sorted(
                   filter(
                       lambda x: '__' not in x, names)
               )] + [''])


def make_toctree(title, names, link):
    """Make a recipe section."""
    return ([f'.. _{link}:',
             '',
             title,
             '-' * len(title),
             '',
             '.. toctree::',
             '']
            + [f'    {name}'
               for name in sorted(
                   filter(
                       lambda x: '__' not in x, names)
               )] + [''])


def make_recipe_documentation(module):
    """Make recipe documentation."""
    mod = importlib.import_module(module)
    members = inspect.getmembers(mod)
    steps = [value for (name, value) in members
             if hasattr(value, '_is_recipe')]

    title = f'{module}'
    rst = [
        f'.. _recipe_{module}:',
        '',
        title,
        '=' * len(title),
        '',
        '',
        '.. contents::',
        '   :local:',
        '',
    ]

    stepnames = [f'{module}@{step.__name__}'
                 if step.__name__ != 'main' else module
                 for step in steps]
    nsteps = len(steps)
    some_steps = 'a single step' if nsteps == 1 else f'{nsteps} steps'
    pyfunclist = [f'  - :py:func:`{module}.{step.__name__}`'
                  for step in steps]
    summary = (['Summary',
                '-------',
                '',
                f'This is the documentation for :py:mod:`{module}`-recipe.',
                f'This recipe is comprised of {some_steps}, namely:',
                '']
               + pyfunclist
               + [
                   '',
                   'Run this recipe through the CLI interface',
                   '',
                   '.. code-block:: console',
                   '',
                   f'   $ asr run {stepnames[-1]}',
                   ''])

    rst.extend(summary)

    if mod.__doc__:
        modrst = ['What does this recipe do?',
                  '-------------------------']
        modrst += mod.__doc__.splitlines()
        rst.extend(modrst)

    for step, stepname in zip(steps, stepnames):
        rst.extend(
            ['',
             stepname,
             '-' * len(stepname),
             f'   .. autofunction:: {module}.{step.__name__}']
        )

    return rst


def generate_api_summary():
    """Generate docs/src/generated/api.rst."""
    rst = ['.. _API reference:',
           '',
           '=============',
           'API reference',
           '=============',
           '',
           '.. contents::',
           '   :local:',
           '']

    for package, title, link, recursive in [
            ('asr', 'Property recipes', 'api recipes', False),
            ('asr/setup', 'Setup recipes', 'api setup recipes', False),
            ('asr/database', 'Database sub-package', 'api database', False),
            ('asr/core', 'Core sub-package', 'api core', True),
            ('asr/test', 'Test sub-package', 'api test', True)]:
        paths = get_modules_from_path(package, recursive=recursive)
        names = get_names_from_paths(paths)
        if paths:
            section = make_section(title=title, names=names, link=link)
            rst.extend(section)

    rst = '\n'.join(rst)
    Path('docs/src/generated/api.rst').write_text(rst)


def get_recipe_modules():
    paths = []
    for package in ['asr', 'asr/setup']:
        paths.extend(get_modules_from_path(package))

    names = get_names_from_paths(paths)
    names = sorted(filter(lambda x: ('__' not in x) and (not x == 'asr.asr'),
                          names))
    return names


def generate_recipe_summary():
    """Generate recipes.rst."""
    rst = ['.. _recipes:',
           '',
           '=================',
           'Available recipes',
           '=================',
           '',
           '.. contents::',
           '   :local:',
           '']

    modules = get_recipe_modules()

    rst.extend(
        ['.. toctree::',
         '   :maxdepth: 1',
         '']
        + [f'   recipe_{module}.rst' for module in modules]
    )
    rst = '\n'.join(rst)
    Path('docs/src/generated/recipes.rst').write_text(rst)


def generate_stub_pages():
    """Generate module stub pages."""
    modules = get_recipe_modules()
    for module in modules:
        rst = make_recipe_documentation(module)
        rst = '\n'.join(rst)
        Path(f'docs/src/generated/recipe_{module}.rst').write_text(rst)


def empty_generated_files():
    """Clear previously generated files."""
    directory = Path('docs/src/generated')
    if not directory.is_dir():
        directory.mkdir()
        return
    paths = directory.glob('*')
    for path in paths:
        path.unlink()
    if not directory.is_dir():
        directory.mkdir()


if __name__ == '__main__':
    empty_generated_files()
    generate_api_summary()
    generate_recipe_summary()
    generate_stub_pages()
