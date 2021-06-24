"""Autogenerate documentation for recipes and modules."""
from typing import get_type_hints
from asr.core import ASRResult, ASRCommand
from pathlib import Path
import importlib
import inspect
import os

CWD = Path('.').absolute()

# pytest-doctest and sphinx-build need different folder setups here.
if CWD.name == 'docs':
    DOCSDIR = CWD
    ROOTDIR = (DOCSDIR / '..').absolute()
elif CWD.name == 'asr':
    DOCSDIR = CWD / 'docs'
    ROOTDIR = CWD
else:
    raise AssertionError(f'Bad CWD={CWD} for documentation generation. '
                         'Only asr/ or asr/docs is allowed.')


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
    return [str(path.absolute().relative_to(ROOTDIR).with_suffix('')).replace('/', '.')
            for path in paths]


def make_section(title, names, link):
    """Make a recipe section."""
    return ([f'.. _{link}:',
             '',
             title,
             '-' * len(title),
             '',
             '.. autosummary::',
             '   :template: autosummary/mytemplate.rst',
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
             if (isinstance(value, ASRCommand) and value.__module__ == module)]

    title = f'{module}'
    rst = [
        f'.. _recipe_{module}:',
        '',
        title,
        '=' * len(title),
        '',
        '',
        '.. contents:: Contents',
        '   :local:',
        '',
    ]

    stepnames = [f'{module}:{step.__name__}'
                 if step.__name__ != 'main' else module
                 for step in steps]
    nsteps = len(steps)
    some_steps = 'a single instruction' if nsteps == 1 else f'{nsteps} instructions'
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
                   '',
                   'or as a python module',
                   '',
                   '.. code-block:: console',
                   '',
                   f'   $ python -m {stepnames[-1]}',
                   ''])

    rst.extend(summary)

    if mod.__doc__:
        modrst = ['Detailed description',
                  '--------------------']
        modrst += mod.__doc__.splitlines()
        rst.extend(modrst)

    rst.extend(['',
                'Steps',
                '-----',
                ''])
    for istep, (step, stepname) in enumerate(zip(steps, stepnames)):
        step_title = f'{stepname}'
        rst.extend(
            ['',
             step_title,
             '^' * len(step_title),
             f'.. autofunction:: {module}.{step.__name__}',
             '   ',
             ]
        )
        if step.returns:
            returns = step.returns
            try:
                if issubclass(returns, ASRResult) and not returns == ASRResult:
                    mod = returns.__module__
                    qualname = returns.__qualname__
                    rst.extend(
                        [
                            f'.. autoclass:: {mod}.{qualname}',
                            '   :members:',
                            '   ',
                        ]
                    )
            except TypeError:
                continue
    return rst


def generate_api_summary():
    """Generate src/generated/api.rst."""
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
            (ROOTDIR / 'asr', 'Property recipes', 'api recipes', False),
            (ROOTDIR / 'asr/setup', 'Setup recipes',
             'api setup recipes', False),
            (ROOTDIR / 'asr/database', 'Database sub-package',
             'api database', False),
            (ROOTDIR / 'asr/core', 'Core sub-package', 'api core', True),
            (ROOTDIR / 'asr/test', 'Test sub-package', 'api test', True)]:
        paths = get_modules_from_path(package, recursive=recursive)
        names = get_names_from_paths(paths)
        if paths:
            section = make_section(title=title, names=names, link=link)
            rst.extend(section)

    rst = '\n'.join(rst)
    write_file('api.rst', rst)


def get_recipe_modules():
    paths = []
    for package in [ROOTDIR / 'asr', ROOTDIR / 'asr/setup']:
        paths.extend(get_modules_from_path(package))

    names = get_names_from_paths(paths)
    names = sorted(filter(lambda x: ('__' not in x) and (not x == 'asr.asr')
                          and (not x == 'asr.setinfo')
                          and (not x == 'asr.setup.materials'),
                          names))
    return names


def get_module_docstring_title(mod):
    mod = importlib.import_module(mod)
    if mod.__doc__:
        return mod.__doc__.split('\n')[0]


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
    modnames = [get_module_docstring_title(mod) for mod in modules]

    rst.extend(
        ['.. toctree::',
         '   :maxdepth: 1',
         '   :hidden:',
         '']
        + [f'   recipe_{module}'
           for module in modules]
    )

    rst.extend(['',
                '.. csv-table::',
                '   :header: "Recipe", "Description"',
                '   :widths: 1, 2',
                ''] +
               [f'   :ref:`{module} <recipe_{module}>`, {modname}'
                if modname else f'   :ref:`{module} <recipe_{module}>`, '
                for modname, module in zip(modnames, modules)])
    rst = '\n'.join(rst)
    write_file('recipes.rst', rst)


def write_file(filename, rst):
    """Write generated rst file."""
    filepath = DOCSDIR / Path(f'src/generated/{filename}')
    print(f'Writing: {filepath}')
    filepath.write_text(rst)


def generate_stub_pages():
    """Generate module stub pages."""
    modules = get_recipe_modules()
    print(f'Found asr modules: {modules}')
    for module in modules:
        rst = make_recipe_documentation(module)
        rst = '\n'.join(rst)
        write_file(f'recipe_{module}.rst', rst)


def empty_generated_files():
    """Clear previously generated files."""
    directory = DOCSDIR / Path('src/generated')
    if not directory.is_dir():
        directory.mkdir()
        return
    paths = directory.glob('*')
    for path in paths:
        path.unlink()
    if not directory.is_dir():
        directory.mkdir()


def update_tutorial_outputs():
    ...


def generate_docs():
    """Generate documentation."""
    print('Generating ASR documentation.')
    empty_generated_files()
    # generate_api_summary()
    generate_recipe_summary()
    generate_stub_pages()
    update_tutorial_outputs()


if __name__ == '__main__':
    generate_docs()
