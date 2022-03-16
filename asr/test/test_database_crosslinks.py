import pytest


@pytest.mark.ci
def test_database_crosslinks(crosslinks_test_dbs):
    """Test asr.database.crosslinks recipe."""
    from ase.db import connect
    from asr.database.crosslinks import main as crosslinks

    # write links to data of 'db.db' file with crosslinks recipe
    crosslinks(target='db.db',
               dbs=['db.db', 'dbref.db'])

    db = connect('db.db')

    reflinks = [
        ('Si2-9552f5fb34d3-Si2', 'testref/testref/Si2-9552f5fb34d3',
            'Example Reference DB'),
        ('BN-BN-d07bd84d0331', 'test/test/BN-d07bd84d0331', 'Example DB'),
        ('Ag-Ag-38f9b4cf2331', 'test/test/Ag-38f9b4cf2331', 'Example DB'),
        ('Fe-Fe-551991cb0ca5', 'test/test/Fe-551991cb0ca5', 'Example DB')]

    reflinks.sort(key=lambda tup: tup[0])

    for row in db.select():
        print(row.data.links)
        links = row.data.links
        links.sort(key=lambda tup: tup[0])
        for i, element in enumerate(links):
            assert element[0] == reflinks[i][0]
            assert element[1] == reflinks[i][1]
            assert element[2] == reflinks[i][2]
