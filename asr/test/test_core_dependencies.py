import pytest
from asr.core.dependencies import Dependencies, Dependency


@pytest.mark.ci
def test_dependencies_append():
    deps = Dependencies()
    dep = Dependency(uid='123', revision='abc')
    deps.append(dep)
    assert dep in deps


@pytest.mark.ci
def test_dependencies_extend():
    deps = Dependencies()
    dep = Dependency(uid='123', revision='abc')
    deps.extend([dep])
    assert dep in deps
