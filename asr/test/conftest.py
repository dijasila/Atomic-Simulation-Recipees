"""Pytest conftest file.

This file configures PyTest. In particular it sets some custom markers
and imports all fixtures which makes them globally available to all
tests.

"""
from ase.parallel import world
from ase.utils import devnull
import pytest
from .fixtures import (mockgpaw, test_material, asr_tmpdir,  # noqa
                       asr_tmpdir_w_params, get_webcontent,  # noqa
                       set_asr_test_environ_variable,  # noqa
                       crosslinks_test_dbs,  # noqa
                       duplicates_test_db)  # noqa


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionstart(session):
    # execute all other hooks to obtain the report object
    config = session.config
    tw = config.get_terminal_writer()
    if world.rank != 0:
        tw._file = devnull
    tr = config.pluginmanager.get_plugin("terminalreporter")
    tr.section('ASR-MPI stuff')
    tr.write(f'size: {world.size}\n')
    yield


def pytest_configure(config):
    """Configure PyTest."""
    # register an additional marker
    markers = [
        "integration_test: Marks an integration test",
        "integration_test_gpaw: Marks an integration test specifically using gpaw",
        "acceptance_test: Marks an acceptance test",
        "ci: Mark a test for running in continuous integration",
        "parallel: Mark a test for parallel testing",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)

    config.addinivalue_line("addopts", "--doctest-modules")
    config.addinivalue_line("filterwarnings", "once::Warning")
    config.addinivalue_line("filterwarnings", "ignore:::matplotlib")
    config.addinivalue_line("filterwarnings",
                            "ignore:numpy.ufunc size changed.*")
    config.addinivalue_line("filterwarnings",
                            "ignore:numpy.dtype size changed.*")
