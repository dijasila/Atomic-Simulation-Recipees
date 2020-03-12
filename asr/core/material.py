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


def material_from_folder(folder='.'):
    """Contruct a material from ASR structure folder.

    Constructs an :ref:`asr.core.material.Material` object from the
    data available in `folder`.

    Parameters
    ----------
    folder : str
        Where to collect material from.
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
