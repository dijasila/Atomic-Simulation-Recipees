"""Defines material class.

A material object closely mimics the behaviour of an ase.db.atomsrow.
"""
from pathlib import Path
from ase.db.row import AtomsRow


def get_row_from_folder(folder='.'):
    """Contruct a material from ASR structure folder.

    Constructs an :class:`asr.core.material.Material` instance from
    the data available in `folder`.

    Parameters
    ----------
    folder : str
        Where to collect material from.

    Returns
    -------
    material : :class:`asr.core.material.Material`
        Output material instance

    """
    from asr.database.fromtree import collect_folder
    from asr.database.browser import RowWrapper
    atoms, kvp, data = collect_folder(
        Path('.'),
        atomsname='structure.json',
        patterns='',
    )
    row = AtomsRow(atoms)
    row.__dict__.update(kvp)
    row._data = data
    row = RowWrapper(row)
    return row


def new_make_panel_figures(context, panels, uid):
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


def make_panel_figures(material, panels, uid=None):
    """Make figures in list of panels.

    Parameters
    ----------
    material : :class:`asr.core.material.Material`
        Material of interest
    panels : list
        List of panels and contents
    Returns
    -------
    None
    """
    pds = []
    for panel in panels:
        pd = panel.get('plot_descriptions', [])
        if pd:
            pds.extend(pd)
            # panel.pop('plot_descriptions')

    for pd in pds:
        if uid is not None:
            filenames = [uid + '-' + filename for filename in pd['filenames']]
        else:
            filenames = pd['filenames']

        pd['function'](material, *filenames)
        figures = ','.join(filenames)
        print(f'Saved figures: {figures}')
