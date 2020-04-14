from .fixtures import (mockgpaw, test_material, asr_tmpdir,  # noqa
                       asr_tmpdir_w_params, get_webcontent)  # noqa


def pytest_configure(config):
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
        """ci: Mark a test for running in continuous integration""",
    )
