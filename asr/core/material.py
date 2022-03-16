"""Defines material class.

A material object closely mimics the behaviour of an ase.db.atomsrow.
"""
import warnings
from pathlib import Path
from ase.db.row import AtomsRow


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
        row = AtomsRow(atoms)
        row.__dict__.update(kvp)
        row._data = data
        self.row = row
        # self.atoms = atoms
        # self.data = data
        # self.kvp = kvp
        # self.cell = atoms.get_cell()
        # self.pbc = atoms.get_pbc()

    def __getattr__(self, key):
        """Wrap row get attribute."""
        return getattr(self.row, key)

    def __contains__(self, key):
        """Is property in key-value-pairs."""
        return key in self.row

    def __iter__(self):
        """Iterate over material attributes."""
        return self.row.__iter__()

    def __getitem__(self, key):
        """Get material attribute."""
        return self.row[key]

    def __setitem__(self, key, value):
        """Set material attribute."""
        self.row[key] = value


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
    from asr.core import decode_object
    from asr.database.fromtree import collect_file
    from ase.io import read
    kvp = {}
    data = {}
    for filename in Path(folder).glob('results-*.json'):
        try:
            tmpkvp, tmpdata = collect_file(filename)
        except ModuleNotFoundError as err:
            # If there are result files named after recipes that are not
            # in the source code, it will trigger import errors.
            # We just warn instead.
            warnings.warn(f'No recipe for resultfile {filename}: {err}')
            continue

        if tmpkvp or tmpdata:
            kvp.update(tmpkvp)
            data.update(tmpdata)

    for key, value in data.items():
        obj = decode_object(value)
        data[key] = obj

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
    return recipe.format_as('ase_webpanel', material, kd)


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
            # panel.pop('plot_descriptions')

    for pd in pds:
        pd['function'](material, *pd['filenames'])
        figures = ','.join(pd['filenames'])
        print(f'Saved figures: {figures}')
