import numpy as np
import pytest
from ase.build import bulk

import asr
from asr.database import connect
from asr.database.database import bytes_to_object, object_to_bytes


@pytest.mark.ci
def test_write_a_record(asr_tmpdir):
    db = connect("database.db")
    ag = bulk("Ag")
    rec = asr.Record(result=2)
    db.write(atoms=ag, records=[rec], data={"some_extra_data.json": {"key": "value"}})

    row = db.get(id=1)
    rec2 = row.records[0]
    assert rec == rec2


@pytest.mark.ci
def test_bytes_to_object_and_back(various_object_types):
    if isinstance(various_object_types, np.ndarray):
        assert all(
            various_object_types
            == bytes_to_object(object_to_bytes(various_object_types))
        )
    else:
        assert various_object_types == bytes_to_object(
            object_to_bytes(various_object_types)
        )
