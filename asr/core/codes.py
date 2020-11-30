import typing
import importlib
from ase.utils import search_current_git_hash


def get_package_version_and_hash(package: str):
    """Get parameter and software version information as a dictionary."""
    mod = importlib.import_module(package)
    githash = search_current_git_hash(mod)
    version = mod.__version__
    return version, githash


class Code:  # noqa

    def __init__(self, package, version, git_hash=None):  # noqa
        self.package = package
        self.version = version
        self.git_hash = git_hash

    @classmethod
    def from_string(cls, package: str):  # noqa
        version, git_hash = get_package_version_and_hash(package)

        return cls(package, version, git_hash)


class Codes:  # noqa

    def __init__(self, codes: typing.List[Code]):  # noqa
        self.codes = codes
