from asr.core.command import FullFeatureFileCache, construct_run_spec, RunRecord


def test_cache(asr_tmpdir):
    run_spec = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    run_record = RunRecord(run_spec=run_spec,
                           result=result)
    cache = FullFeatureFileCache()

    assert not cache.has(run_spec)
    cache.add(run_record)
    assert cache.has(run_spec)
    assert cache.get(run_spec) == run_spec
