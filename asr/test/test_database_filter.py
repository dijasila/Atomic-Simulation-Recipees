import pytest


@pytest.fixture
def make_test_database(separate_folder):
    from ase.db import connect
    from .conftest import BN, Ag, Fe
    con = connect('database.db')
    con.write(atoms=BN, tmpid=1,
              data={'results-asr.gs.json': {'content': 123}})
    con.write(atoms=Fe, tmpid=2,
              data={'results-asr.relax.json': {'content': 321}})
    con.write(atoms=Ag, tmpid=3)
    con.metadata = {'keys': ['onekey', 'anotherkey']}
    return con


@pytest.mark.parametrize('selection,patterns',
                         [('BN', '*gs*'),
                          ('BN', '*gs*,*relax*'),
                          ('Fe', '')])
def test_database_filter(make_test_database, selection, patterns):
    from asr.database.filter import main as dbfilter
    from ase.db import connect
    from fnmatch import fnmatch
    filterdb = 'filtered.db'
    dbfilter('database.db', filterdb, selection=selection, patterns=patterns)

    con = make_test_database
    assert con.metadata
    con_filter = connect(filterdb)
    assert con_filter.metadata == con.metadata
    con_rows = list(con.select(selection))
    con_filter_rows = list(con_filter.select())
    assert len(con_rows) == len(con_filter_rows)
    for row in con_filter_rows:
        for key in row.data:
            assert any([fnmatch(key, pattern) for pattern in patterns])
        row2 = con.get(tmpid=row.tmpid)
        assert row.toatoms() == row2.toatoms()
