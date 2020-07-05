"""Implement cache functionality."""


class ASRCache:
    """Object to handle reading of caches."""

    def __init__(self, filename):
        """Initialize cache using filename."""
        self.filename = filename

    def get(self, *args, **kwargs):
        """Get a cache entry based on args and kwargs."""
        pass

    def has(self, *args, **kwargs):
        """Check if a cache entry matching args and kwargs exists."""
        pass

    def add(self, results, argtuple):
        """Add another cache entry."""
        args, kwargs = args
