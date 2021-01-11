import pytest
from asr.core.record import RunRecord
from asr.core.specification import construct_run_spec
from asr.core.params import Parameters


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
def test_record_to_string(data, result):
    rec = RunRecord(**data)
    string = str(rec)

    assert string == result


@pytest.mark.ci
@pytest.mark.parametrize(
    'record1,record2,is_equal',
    [
        (
            RunRecord(result=5),
            RunRecord(result=5),
            True,
        ),
        (
            RunRecord(result=4),
            RunRecord(result=5),
            False,
        ),
        (
            RunRecord(result=5.0),
            RunRecord(result=5),
            True,
        )
    ]

)
def test_record_equality(record1, record2, is_equal):
    assert (record1 == record2) is is_equal


@pytest.fixture()
def record():
    rec = RunRecord(result=5)
    return rec


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
@pytest.mark.parametrize('attr', ['name', 'parameters', 'uid'])
def test_record_get_property(attr):
    data = dict(
        parameters=Parameters({'a': 1}),
        name='asr.test',
        uid='myveryspecialuid',
    )
    spec = construct_run_spec(**data)
    rec = RunRecord(run_specification=spec)
    assert getattr(rec, attr) == data[attr]
