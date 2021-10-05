from asr.database.project import make_project_from_config

def test_database_project():

    no_grouping = lambda x, y: False

    row_generator = [BN, graphene]
    make_key_value_pairs = lambda data: dict(natoms=len(data))
    project = make_project_from_config(
        grouping=no_grouping,
        get_rows=row_generator,     
        make_kvp=make_key_value_pairs,
    )

    assert project.grouping == no_grouping
    
    db = project.collect_database()

    assert len(db)