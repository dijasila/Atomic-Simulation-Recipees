"""Defines material class.

A material object closely mimics the behaviour of an ase.db.atomsrow.
"""
from pathlib import Path
from ase.db.row import AtomsRow


def make_panel_figures(context, panels, uid):
    from ase.utils import workdir
    paths = []
    for panel in panels:
        for plot_description in panel.get('plot_descriptions', []):
            path = Path(f'fig-{context.name}-{uid}')
            with workdir(path, mkdir=True):
                func = plot_description['function']
                filenames = plot_description['filenames']
                import inspect
                argspec = inspect.getargspec(func)

                if argspec.args[0] == 'context':
                    func(context, *filenames)
                else:
                    func(context.row, *filenames)

            for filename in filenames:
                paths.append(path / filename)

    return paths
