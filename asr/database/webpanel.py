from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol


class Content(Protocol):
    ...


@dataclass
class TwoColumns:
    """Represents a content in a column format."""

    column1: List[Content] = field(default_factory=lambda: [])
    column2: List[Content] = field(default_factory=lambda: [])

    def __getitem__(self, index):
        return [self.column1, self.column2][index]

    def __setitem__(self, index, value):
        [self.column1, self.column2][index] = value

    def __len__(self):
        return 2


@dataclass
class Figure:
    """Class that represents a figure."""

    function: Callable
    filenames: List[str]

    # XXX getitem only required for compatibility with
    # existing webpanels.
    def __getitem__(self, index):
        return self.__dict__[index]


@dataclass
class Table:
    """Class that represents a table."""

    header: List[str]
    rows: List[List[str]]
    columnwidth: int = 3


@dataclass
class DescribedContent:

    content: Content
    description: str
    title: str = "Information"

    def __str__(self):
        return str(self.content)

    def __hash__(self):
        return hash(self.content)


@dataclass
class WebPanel(Mapping):

    title: str
    columns: List[List] = field(default_factory=lambda: [[], []])
    plot_descriptions: List[Figure] = field(default_factory=lambda: [])
    sort: int = 99
    id: Optional[str] = None

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        return iter(self.__dict__)


@dataclass
class Panel:

    title: str
    columns: List[List] = field(default_factory=lambda: [[], []])
    plot_descriptions: List[Figure] = field(default_factory=lambda: [])
    sort: int = 99
    id: Optional[str] = None

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        return iter(self.__dict__)
