"""General purpose utilities."""
from datetime import datetime

_LATEST_PRINT = None


def timed_print(*args, wait=20):
    """Print at most every `wait` seconds."""
    global _LATEST_PRINT
    now = datetime.now()
    if _LATEST_PRINT is None or (now - _LATEST_PRINT).seconds > wait:
        print(*args)
        _LATEST_PRINT = now
