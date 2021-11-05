from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Figure:
    """Class that represents a figure."""

    function: Callable
    filenames: List[str]


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
