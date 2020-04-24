import pytest


@pytest.mark.ci
def test_env_variables():
    """Check that the test environment variable is set."""
    import os
    assert os.environ['ASRTESTENV']
