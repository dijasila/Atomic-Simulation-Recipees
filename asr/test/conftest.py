from .fixtures import (fast_params, mockgpaw, test_material,  # noqa
                       asr_tmpdir_w_params)  # noqa


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
