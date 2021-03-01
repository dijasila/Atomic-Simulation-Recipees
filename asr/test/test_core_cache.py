import pytest

from asr.core.cache import Cache, get_cache
from asr.core.specification import construct_run_spec
from asr.core.record import Record


@pytest.mark.ci
def test_cache(cache):
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    run_record = Record(run_specification=run_spec,
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
    run_record = Record(run_specification=run_spec,
                        result={'b': 1})

    assert not cache.has(run_specification=run_spec)
    cache.add(run_record)
    assert cache.has(run_specification=run_spec)
    assert cache.get(run_specification=run_spec) == run_record


@pytest.fixture(params=['filesystem', 'memory'])
def cache(request, asr_tmpdir):
    return get_cache(request.param)


@pytest.mark.ci
def test_cache_has(cache, record):
    cache.add(record)
    assert cache.has(**record)


@pytest.mark.ci
def test_cache_contains(cache, record):
    cache.add(record)
    assert record in cache


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
    other.run_specification.uid = 'someotheruid'
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


@pytest.mark.ci
def test_cache_remove(cache, record):
    other = record.copy()
    other.run_specification.uid = 'someotheruid'

    cache.add(record)
    cache.add(other)

    assert len(cache.select()) == 2

    removed_records = cache.remove(uid='someotheruid')

    records = cache.select()

    assert len(records) == 1
    assert records[0] == record
    assert len(removed_records) == 1
    assert removed_records[0] == other


def add_record(cache, record):
    cache.add(record)


@pytest.mark.ci
def test_cache_add_concurrent_processes(asr_tmpdir):
    import multiprocessing
    from asr.core.specification import get_new_uuid
    cache = get_cache('filesystem')
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    record = Record(
        run_specification=run_spec,
        result=1,
    )

    nprocs = 10

    processes = []
    records = []
    for jobid in range(nprocs):
        record = record.copy()
        uid = get_new_uuid()
        record.run_specification.uid = uid
        records.append(record)
        proc = multiprocessing.Process(
            target=add_record,
            args=(cache, record))
        processes.append(proc)
        proc.start()

    for procid, proc in enumerate(processes):
        proc.join()
        assert proc.exitcode == 0, f'Error in Job #{procid}.'

    for record in records:
        assert cache.has(uid=record.uid)
