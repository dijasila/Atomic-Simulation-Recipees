"""Module that exists primarily to satisfy mypy.

Implements a Dummy "history" base class that the RevisionHistory
can inherit from and we can this satisfy mypy.
"""
import abc
from typing import Any


class History(abc.ABC):
    """Base class for revision history."""

    @abc.abstractmethod
    def add(self, revision: Any) -> None:
        pass
