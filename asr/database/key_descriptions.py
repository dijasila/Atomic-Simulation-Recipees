from typing import Optional
from dataclasses import dataclass


@dataclass
class KeyDescription:
    """A class that represents a key description.

    Attributes
    ----------
    short : str
        A short description of the key.
    long : str
        A long description of the key.
    unit : str
        The unit of the key.
    """

    short: str
    long: str
    unit: str


def make_key_description(
    short: str,
    long: Optional[str] = None,
    unit: str = "",
) -> KeyDescription:
    """Make a key description.

    Parameters
    ----------
    short : str
        The short description of the key.
    long : Optional[str], optional
        The long description of the key, by default None. If "None", this defaults
        to the short description.
    unit : str, optional
        The unit of the key, by default "" meaning "no unit".

    Returns
    -------
    KeyDescription
    """
    if long is None:
        long = short
    return KeyDescription(short, long, unit)
