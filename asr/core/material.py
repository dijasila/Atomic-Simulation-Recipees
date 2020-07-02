"""Defines material class.

A material object closely mimics the behaviour of an ase.db.atomsrow.
"""
from pathlib import Path


class Material:
    def __init__(self, atoms, kvp, data):
        """Construct material object.

        Make make material instance. This objects bind together an
        atomic structure with its key-value-pairs and its raw data and
        closely mimics the structure of an ase.db.atomsrow instance.

        Parameters
        ----------
        atoms : ase.Atoms object
        kvp : dict
            Key value pairs associated with atomic stucture.
        data : dict
            Raw data associated with atomic structure-

        """
        self.__dict__.update(kvp)
        self.atoms = atoms
        self.data = data
        self.kvp = kvp
        self.cell = atoms.get_cell()
        self.pbc = atoms.get_pbc()

    def __contains__(self, key):
        """Is property in key-value-pairs."""
        return key in self.kvp

    def __iter__(self):
        """Iterate over material attributes."""
        return (key for key in self.__dict__ if key[0] != '_')

    def __getitem__(self, key):
        """Get material attribute."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set material attribute."""
        setattr(self, key, value)

    def get(self, key, default=None):
        return self.kvp.get(key, default)

    def toatoms(self):
        return self.atoms


def get_material_from_folder(folder='.'):
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
    from asr.database.fromtree import collect_file
    from ase.io import read
    kvp = {}
    data = {}
    for filename in Path(folder).glob('results-*.json'):
        tmpkvp, tmpdata = collect_file(filename)
        if tmpkvp or tmpdata:
            kvp.update(tmpkvp)
            data.update(tmpdata)

    atoms = read('structure.json', parallel=False)
    material = Material(atoms, kvp, data)

    return material


def get_webpanels_from_material(material, recipe):
    """Return web-panels of recipe.

    Parameters
    ----------
    material : :class:`asr.core.material.Material`
        Material on which the webpanel should be evaluated
    recipe : :class:`asr.core.ASRCommand`
        Recipe instance

    Returns
    -------
    panels : list
        List of panels and contents.
    """
    from asr.database.app import create_key_descriptions
    kd = create_key_descriptions()
    return recipe.webpanel(material, kd)


def make_panel_figures(material, panels):
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
            panel.pop('plot_descriptions')

    for pd in pds:
        pd['function'](material, *pd['filenames'])
        figures = ','.join(pd['filenames'])
        print(f'Saved figures: {figures}')
