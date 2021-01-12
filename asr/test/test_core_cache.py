import pytest

from asr.core.cache import Cache, get_cache
from asr.core.specification import construct_run_spec
from asr.core.record import RunRecord


@pytest.mark.ci
def test_cache(cache):
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    run_record = RunRecord(run_specification=run_spec,
                           result={'b': 1})

    assert not cache.has(run_specification=run_spec)
    cache.add(run_record)
    assert cache.has(run_specification=run_spec)
    assert cache.get(run_specification=run_spec) == run_record
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 2},
        version=0,
    )
    run_record = RunRecord(run_specification=run_spec,
                           result={'b': 1})

    assert not cache.has(run_specification=run_spec)
    cache.add(run_record)
    assert cache.has(run_specification=run_spec)
    assert cache.get(run_specification=run_spec) == run_record


@pytest.fixture(params=['filesystem', 'memory'])
def cache(request, asr_tmpdir):
    return get_cache(request.param)


@pytest.fixture
def record(various_object_types):
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    run_record = RunRecord(
        run_specification=run_spec,
        result=various_object_types,
    )
    return run_record


@pytest.mark.ci
def test_cache_has(cache, record):
    cache.add(record)
    assert cache.has(**record)


@pytest.mark.ci
def test_cache_has_dont_have(cache, record):
    cache.add(record)
    assert not cache.has(**{'run_specification.uid': 0})


@pytest.mark.ci
def test_cache_add(cache, record):
    run_spec = record.run_specification
    assert not cache.has(run_specification=run_spec)

    cache.add(record)

    assert cache.has(run_specification=run_spec)
    fetched_record = cache.get(**record)

    if isinstance(record.result, tuple):
        pytest.xfail(reason='JSONSerializer cannot distinguish tuple. '
                     'They are always cast to lists.')

    assert record == fetched_record


@pytest.mark.ci
def test_cache_add_raises_when_adding_duplicate_records(cache, record):
    cache.add(record)

    with pytest.raises(AssertionError):
        cache.add(record)


@pytest.mark.ci
def test_cache_update(cache, record):
    cache.add(record)
    uid = record.uid
    updated_record = record.copy()
    updated_record.result = 'Updated result'
    cache.update(updated_record)
    fetched_record = cache.get(uid=uid)

    assert updated_record == fetched_record
    assert updated_record != record


@pytest.mark.ci
def test_cache_get(cache, record):
    cache.add(record)
    fetched_record = cache.get(**record)

    if isinstance(record.result, tuple):
        pytest.xfail(reason='JSONSerializer cannot distinguish tuple. '
                     'They are always cast to lists.')
    assert fetched_record == record


@pytest.mark.ci
def test_cache_get_raises_when_getting_multiple_records(cache, record):
    other = record.copy()
    other.run_specification.uid = 0
    cache.add(record)
    cache.add(other)

    with pytest.raises(AssertionError):
        cache.get()


@pytest.mark.ci
def test_cache_select(cache, record):
    cache.add(record)
    fetched_records = cache.select(**record)

    assert len(fetched_records) == 1


@pytest.mark.ci
@pytest.mark.parametrize('backend', ['filesystem', 'memory'])
def test_get_cache(backend):
    cache = get_cache(backend=backend)
    assert isinstance(cache, Cache)
