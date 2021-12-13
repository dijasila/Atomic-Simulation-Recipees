from functools import wraps
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import simplejson as json
from ase import Atoms
from ase.db import connect as ase_connect
from ase.db.core import Database
from ase.db.row import AtomsRow
from ase.io.jsonio import create_ase_object

from asr.core.serialize import dict_to_object, object_to_dict


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
        return self.data.get("records", [])

    @property
    def cache(self):
        from asr import get_cache

        mem_cache = get_cache("memory")
        for record in self.records:
            mem_cache.add(record)
        return mem_cache

    def _load_data(self):
        rowdata = self.row._data
        _data = bytes_to_object(rowdata)
        if "records" in _data:
            assert isinstance(_data["records"], list)
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

    @property
    def cell(self):
        return self.row.cell

    @property
    def pbc(self):
        return self.row.pbc

    def __getattr__(self, name):
        return getattr(self.row, name)


class ASEDatabaseInterface:
    """Interface for ASE database.

    Ensures correct data serialization.
    """

    def __init__(self, db: Database):
        self.db = db

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.db.metadata

    @metadata.setter
    def metadata(self, value):
        self.db.metadata = value

    @wraps(Database.write)
    def write(self, *args, data=None, records=None, **kwargs):
        if data is None:
            data = {}
        container = {**data}

        if records:
            container["records"] = records

        bts = object_to_bytes(container)

        assert isinstance(bts, bytes)
        return self.db.write(data=bts, *args, **kwargs)

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
    db = ase_connect(dbname, serial=True)
    return ASEDatabaseInterface(db)


def object_to_bytes(obj: Any) -> bytes:
    """Serialize Python object to bytes."""
    parts = [b"12345678"]
    obj = o2b(obj, parts)
    offset = sum(len(part) for part in parts)
    x = np.array(offset, np.int64)
    if not np.little_endian:
        x.byteswap(True)
    parts[0] = x.tobytes()
    parts.append(json.dumps(obj, separators=(",", ":"), tuple_as_array=False).encode())
    bts = b"".join(parts)
    return bts


def bytes_to_object(b: bytes) -> Any:
    """Deserialize bytes to Python object."""
    x = np.frombuffer(b[:8], np.int64)
    if not np.little_endian:
        x = x.byteswap()
    offset = x.item()
    obj = json.loads(b[offset:].decode())
    bts = b2o(obj, b)
    return bts


def o2b(obj: Any, parts: List[bytes]):
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    if isinstance(obj, dict):
        return {key: o2b(value, parts) for key, value in obj.items()}
    if isinstance(obj, list):
        return [o2b(value, parts) for value in obj]
    if isinstance(obj, tuple):
        return {"__type__": "tuple", "value": [o2b(value, parts) for value in obj]}
    if isinstance(obj, np.ndarray):
        assert obj.dtype != object, 'Cannot convert ndarray of type "object" to bytes.'
        offset = sum(len(part) for part in parts)
        if not np.little_endian:
            obj = obj.byteswap()
        parts.append(obj.tobytes())
        return {"__ndarray__": [list(obj.shape), obj.dtype.name, offset]}
    if isinstance(obj, complex):
        return {"__complex__": [obj.real, obj.imag]}
    try:
        objtype = getattr(obj, "ase_objtype")
        if objtype:
            dct = o2b(obj.todict(), parts)
            dct["__ase_objtype__"] = objtype
            return dct
    except AttributeError:
        # Fall back to ASRs other custom serializations
        dct = o2b(object_to_dict(obj), parts)
        return dct
    raise ValueError("Objects of type {type} not allowed".format(type=type(obj)))


def b2o(obj: Any, b: bytes) -> Any:
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj

    if isinstance(obj, list):
        return [b2o(value, b) for value in obj]

    assert isinstance(obj, dict)

    tp = obj.get("__type__")

    if tp == "tuple":
        return tuple(b2o(value, b) for value in obj["value"])

    x = obj.get("__complex__")
    if x is not None:
        return complex(*x)

    x = obj.get("__ndarray__")
    if x is not None:
        shape, name, offset = x
        dtype = np.dtype(name)
        size = dtype.itemsize * np.prod(shape).astype(int)
        a = np.frombuffer(b[offset : offset + size], dtype)
        a.shape = shape
        if not np.little_endian:
            a = a.byteswap()
        return a

    dct = {key: b2o(value, b) for key, value in obj.items()}
    objtype = dct.pop("__ase_objtype__", None)
    if objtype is not None:
        return create_ase_object(objtype, dct)
    try:
        # See if this is another custom type
        return dict_to_object(dct)
    except ValueError:
        return dct
