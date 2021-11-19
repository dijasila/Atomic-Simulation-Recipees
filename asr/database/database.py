from functools import wraps
from typing import Generator, Tuple

from ase import Atoms
from ase.db import connect as ase_connect
from ase.db.core import Database
from ase.db.row import AtomsRow


class Row:
    """ASE AtomsRow wrapper.

    Ensures correct data (de)-serialization.
    """

    def __init__(self, row: AtomsRow):
        self.row = row
        self._data = None

    def __contains__(self, *args, **kwargs):
        return self.row.__contains__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        yield from self.row.__iter__(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self.row.get(*args, **kwargs)

    def count_atoms(self, *args, **kwargs):
        return self.row.count_atoms(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.row.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.row.__setitem__(*args, **kwargs)

    def __str__(self, *args, **kwargs):
        return self.row.__str__(*args, **kwargs)

    def toatoms(self, *args, **kwargs):
        return self.row.toatoms(*args, **kwargs)

    @property
    def key_value_pairs(self):
        return self.row.key_value_pairs

    @property
    def constraints(self):
        return self.row.constraints

    @property
    def data(self):
        if self._data is None:
            self._load_data()
        return self._data

    @property
    def records(self):
        if self._data is None:
            self._load_data()
        return self._data["records"]

    @property
    def cache(self):
        from asr import get_cache
        mem_cache = get_cache("memory")
        for record in self.records:
            mem_cache.add(record)
        return mem_cache

    def _load_data(self):
        from .fromtree import serializer
        rowdata = self.row.data
        _data = {key: value for key, value in rowdata.items() if key != "records"}

        if "records" in rowdata:
            records = serializer.deserialize(rowdata["records"])
            _data["records"] = records
        self._data = _data

    @property
    def natoms(self):
        return self.row.natoms

    @property
    def formula(self):
        return self.row.formula

    @property
    def symbols(self):
        return self.row.symbols

    @property
    def fmax(self):
        return self.row.fmax

    @property
    def contrained_forces(self):
        return self.row.contrained_forces

    @property
    def smax(self):
        return self.row.smax

    @property
    def mass(self):
        return self.row.mass

    @property
    def volume(self):
        return self.row.volume

    @property
    def charge(self):
        return self.row.charge

    @property
    def id(self):
        return self.row.id

    def __getattr__(self, name):
        return self.key_value_pairs[name]


class ASEDatabaseInterface:
    """Interface for ASE database.

    Ensures correct data serialization.
    """

    def __init__(self, db: Database):
        self.db = db

    @wraps(Database.metadata)
    def metadata(self, *args, **kwargs):
        return self.db.metadata(*args, **kwargs)

    @wraps(Database.write)
    def write(self, *args, data=None, records=None, **kwargs):
        from .fromtree import serializer

        if data is None:
            data = {}
        container = {**data}

        if records:
            container["records"] = serializer.serialize(records)

        return self.db.write(data=container, *args, **kwargs)

    @wraps(Database.reserve)
    def reserve(self, *args, **kwargs):
        return self.db.reserve(*args, **kwargs)

    @wraps(Database.get_atoms)
    def get_atoms(self, *args, **kwargs) -> Atoms:
        return self.db.get_atoms(*args, **kwargs)

    @wraps(Database.get)
    def get(self, *args, **kwargs) -> Row:
        row = self.db.get(*args, **kwargs)
        return Row(row)

    @wraps(Database.select)
    def select(self, *args, **kwargs) -> Generator[Row, None, None]:
        for row in self.db.select(*args, **kwargs):
            yield Row(row)

    @wraps(Database.count)
    def count(self, *args, **kwargs) -> int:
        return self.db.count(*args, **kwargs)

    @wraps(Database.__len__)
    def __len__(self, *args, **kwargs) -> int:
        return self.db.__len__(*args, **kwargs)

    @wraps(Database.update)
    def update(self, *args, **kwargs) -> Tuple[int, int]:
        return self.db.update(*args, **kwargs)

    @wraps(Database.delete)
    def delete(self, *args, **kwargs):
        return self.db.delete(*args, **kwargs)

    @wraps(Database.__getitem__)
    def __getitem__(self, *args, **kwargs):
        row = self.db.__getitem__(*args, **kwargs)
        return Row(row)

    @wraps(Database.__delitem__)
    def __delitem__(self, *args, **kwargs):
        return self.db.__detitem__(*args, **kwargs)

    def __enter__(self):
        self.db.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.db.__exit__(*args, **kwargs)


def connect(dbname: str) -> ASEDatabaseInterface:
    """Connect to database.

    Connects to an ASE database and provides a useful
    interface that takes care of all data serialization
    and de-serialization.

    Parameters
    ----------
    dbname : str
        Name of database

    Returns
    -------
    ASEDatabaseInterface
        ASR-ASE database interface.
    """
    db = ase_connect(dbname)
    return ASEDatabaseInterface(db)
