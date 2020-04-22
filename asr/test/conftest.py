"""Pytest conftest file.

This file configures PyTest. In particular it sets some custom markers
and imports all fixtures which makes them globally available to all
tests.

"""
from ase.parallel import world
from ase.utils import devnull
import pytest
from .fixtures import (mockgpaw, test_material, asr_tmpdir,  # noqa
                       asr_tmpdir_w_params, get_webcontent)  # noqa


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
    config.addinivalue_line(
        "markers",
        """integration_test: Marks an integration test""",
    )
    config.addinivalue_line(
        "markers",
        """integration_test_gpaw: Marks an integration
        test specifically using gpaw""",
    )
    config.addinivalue_line(
        "markers",
        """acceptance_test: Marks an acceptance test""",
    )

    config.addinivalue_line(
        "markers",
        """ci: Mark a test for running in continuous integration"""
    )

    config.addinivalue_line(
        "markers",
        """parallel: Mark a test for parallel testing"""
    )
