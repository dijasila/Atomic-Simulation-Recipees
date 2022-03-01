import pytest
import copy
from asr.core.runner import runner
from asr.core.specification import construct_run_spec


def do_nothing(*args, **kwargs):
    return args, kwargs


@pytest.fixture()
def run_spec():
    return construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )


@pytest.mark.xfail(reason='now now')
@pytest.mark.ci
def test_runner(asr_tmpdir, run_spec):
    result = runner.run(do_nothing, run_spec)
    assert result == ((run_spec, ), {})


def target(run_spec):
    result = runner.run(do_nothing, run_spec)
    assert result == ((run_spec, ), {})


@pytest.mark.xfail(reason='now now')
@pytest.mark.ci
def test_runner_concurrent_processes_asking_for_workdir_simultaneously(asr_tmpdir):
    import multiprocessing
    from asr.core.specification import get_new_uuid
    run_specification = construct_run_spec(
        name='asr.test',
        parameters={'a': 1},
        version=0,
    )
    nprocs = 10

    processes = []
    for jobid in range(nprocs):
        run_spec = copy.deepcopy(run_specification)
        uid = get_new_uuid()
        run_spec.uid = uid
        proc = multiprocessing.Process(
            target=target,
            args=(run_spec, ))
        processes.append(proc)
        proc.start()

    for procid, proc in enumerate(processes):
        proc.join()
        assert proc.exitcode == 0, f'Error in Job #{procid}.'
