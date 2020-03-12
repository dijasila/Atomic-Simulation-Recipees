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
        self.atoms = atoms
        self.data = data
        self.kvp = kvp
        self.cell = atoms.get_cell()
        self.pbc = atoms.get_pbc()

    def get(self, key, default=None):
        return self.kvp.get(key, default)

    def __getattr__(self, key):
        if key == "data":
            return self.data
        return self.kvp[key]

    def toatoms(self):
        return self.atoms


def get_material_from_folder(folder='.'):
    """Contruct a material from ASR structure folder.

    Constructs an :ref:`asr.core.material.Material` object from the
    data available in `folder`.

    Parameters
    ----------
    folder : str
        Where to collect material from.

    Returns
    -------
    material : :ref:`asr.core.material.Material`-instance
        Output material instance
    """
    from asr.database.fromtree import collect
    from ase.io import read
    kvp = {}
    data = {}
    for filename in Path(folder).glob('results-*.json'):
        tmpkvp, tmpkd, tmpdata, tmplinks = collect(str(filename))
        if tmpkvp or tmpkd or tmpdata or tmplinks:
            kvp.update(tmpkvp)
            data.update(tmpdata)

    atoms = read('structure.json', parallel=False)
    material = Material(atoms, kvp, data)

    return material


def get_webpanels_from_material(material, recipe):
    """Return web-panels of recipe.

    Parameters
    ----------
    material : :ref:`asr.core.material.Material`-instance
        Material on which the webpanel should be evaluated
    recipe : :ref:`asr.core.__init__.ASRCommand`-instance
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
    material : :ref:`asr.core.material.Material`-instance
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
