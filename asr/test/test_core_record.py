import pytest
from asr.core.record import Record
from asr.core.specification import construct_run_spec
from asr.core.parameters import Parameters


@pytest.mark.ci
@pytest.mark.parametrize(
    'data,result',
    [
        ({'result': 4}, 'Record(result=4)'),
        (
            {'result': 'verylongstring' * 20},
            'Record(result=verylongstringverylongstringve...)'
        ),
    ]
)
def test_record_repr(data, result):
    rec = Record(**data)
    string = repr(rec)

    assert string == result


@pytest.mark.ci
def test_record_str(record):
    assert 'name=asr.test' in str(record)


@pytest.mark.ci
@pytest.mark.parametrize(
    'record1,record2,is_equal',
    [
        (
            Record(result=5),
            Record(result=5),
            True,
        ),
        (
            Record(result=4),
            Record(result=5),
            False,
        ),
        (
            Record(result=5.0),
            Record(result=5),
            True,
        )
    ]

)
def test_record_equality(record1, record2, is_equal):
    assert (record1 == record2) is is_equal


@pytest.fixture()
def record():
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    run_record = Record(
        run_specification=run_spec,
        result=5,
    )
    return run_record


@pytest.fixture()
def copied_record(record):
    rec2 = record.copy()
    return record, rec2


@pytest.mark.ci
def test_record_copy_is_equal(copied_record):
    rec, rec2 = copied_record
    assert rec == rec2


@pytest.mark.ci
def test_record_copy_is_not_same_object(copied_record):
    rec, rec2 = copied_record
    assert rec is not rec2


@pytest.mark.ci
def test_record_copy_setting_result(copied_record):
    rec, rec2 = copied_record
    rec2.result = 'Some random result'
    assert rec.result != rec2.result


@pytest.mark.ci
def test_record_copy_setting_uid(copied_record):
    rec, rec2 = copied_record
    rec2.run_specification.uid = 'RandomUID'
    assert rec.run_specification.uid != rec2.run_specification.uid


@pytest.mark.ci
@pytest.mark.parametrize('attr', ['name', 'parameters', 'uid'])
def test_record_get_property(attr):
    data = dict(
        parameters=Parameters({'a': 1}),
        name='asr.test',
        uid='myveryspecialuid',
    )
    spec = construct_run_spec(**data)
    rec = Record(run_specification=spec)
    assert getattr(rec, attr) == data[attr]
