import textwrap
import typing
import importlib
from dataclasses import dataclass


def get_package_version_and_hash(package: str):
    """Get parameter and software version information as a dictionary."""
    from ase.utils import search_current_git_hash
    mod = importlib.import_module(package)
    githash = search_current_git_hash(mod)
    version = mod.__version__
    return version, githash


@dataclass
class Code:

    package: typing.Optional[str] = None
    version: typing.Optional[str] = None
    git_hash: typing.Optional[str] = None

    @classmethod
    def from_string(cls, package: str):  # noqa
        version, git_hash = get_package_version_and_hash(package)

        return cls(package, version, git_hash)

    def __str__(self):
        lines = []
        for key, value in sorted(self.__dict__.items(), key=lambda item: item[0]):
            value = str(value)
            if '\n' in value:
                value = '\n' + textwrap.indent(value, ' ')
            lines.append(f'{key}={value}')
        return '\n'.join(lines)

    def __eq__(self, other):
        if not isinstance(other, Code):
            return False
        return self.__dict__ == other.__dict__


@dataclass
class Codes:
    codes: typing.List[Code]

    def __str__(self):
        lines = []
        for code in sorted(self.codes, key=lambda item: item.package):
            value = str(code)
            if '\n' in value:
                value = '\n' + textwrap.indent(value, ' ')
            lines.append(f'code={value}')
        return '\n'.join(lines)

    def __eq__(self, other):
        if not isinstance(other, Codes):
            return False
        return self.codes == other.codes
