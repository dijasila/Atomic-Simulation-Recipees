"""Module that exists primarily to satisfy mypy.

Implements a Dummy "history" base class that the RevisionHistory
can inherit from and we can this satisfy mypy.
"""


class History:
    """Base class for revision history."""
