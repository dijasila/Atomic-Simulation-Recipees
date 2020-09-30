"""Autogenerate documentation for recipes and modules."""
from pathlib import Path
import importlib
import inspect


def get_modules_from_path(path: str):
    """Get modules from path."""
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
             '   :template: autosummary/recipetemplate.rst',
             '   :toctree: generated',
             ''] +
            [f'    {name}'
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
             ''] +
            [f'    generated/{name}'
             for name in sorted(
                     filter(
                         lambda x: '__' not in x, names)
             )] + [''])


def make_recipe_documentation(module):
    """Make recipe documentation."""
    mod = importlib.import_module(module)

    members = inspect.getmembers(mod)

    functions = filter(
        lambda member: inspect.getmodule(member) == mod,
        filter(
            inspect.isfunction,
            (member for (name, member) in members)
        )
    )
    rst = [module,
           '=' * len(module)]
    for function in functions:
        rst.append(f'.. autofunction:: {module}.{function.__name__}')

    return rst


def generate_api_summary():
    """Generate docs/src/api.rst."""
    rst = ['.. _API reference:',
           '',
           '=============',
           'API reference',
           '=============',
           '',
           '.. contents::',
           '   :local:',
           '']

    for package, title, link in [
            ('asr', 'Property recipes', 'api recipes'),
            ('asr/setup', 'Setup recipes', 'api setup recipes'),
            ('asr/database', 'Database sub-package', 'api database'),
            ('asr/core', 'Core sub-package', 'api core'),
            ('asr/test', 'Test sub-package', 'api test')]:
        paths = get_modules_from_path(package)
        names = get_names_from_paths(paths)
        if paths:
            section = make_toctree(title=title, names=names, link=link)
            rst.extend(section)

    rst = '\n'.join(rst)
    Path('docs/src/api.rst').write_text(rst)


def generate_stub_pages():
    """Generate module stub pages."""
    for package in ['asr',
                    'asr/setup',
                    'asr/database',
                    'asr/core',
                    'asr/test']:
        paths = get_modules_from_path(package)
        modules = get_names_from_paths(paths)
        for module in modules:
            rst = make_recipe_documentation(module)
            rst = '\n'.join(rst)
            Path(f'docs/src/generated/{module}.rst').write_text(rst)


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
    generate_stub_pages()
