"""Implements ASRResults object and related quantities."""
from typing import get_type_hints, List


def get_object_descriptions(obj):

    return obj.descriptions


def get_object_types(obj):
    return get_type_hints(obj)


def format_key_description_pair(key: str, attr_type: type, description: str):
    return f'{key}: {attr_type}\n    {description}'


def set_docstring(obj):
    descriptions = get_object_descriptions(obj)
    types = get_object_types(obj)
    assert set(types) == set(descriptions)
    docstring_parts: List[str] = []
    for key in descriptions:
        description = descriptions[key]
        attr_type = types[key]
        string = format_key_description_pair(key, attr_type, description)
        docstring_parts.append(string)

    return "\n".join(docstring_parts)


class WebPanel:
    """Web-panel for presenting results."""

    pass


class ASRResults:
    """Base class for describing results generated with recipes.

    WIP: Over time, by default, this class should be immutable.

    Attributes
    ----------
    version : int
        The version number.
    """

    version: int = 0

    def __init__(self, dct,
                 webpanel: WebPanel):
        """Initialize results from dict."""
        self._check_dct(dct)
        self._dct = dct
        self._webpanel = webpanel

    def __getitem__(self, item):
        """Get item from self.dct."""
        return self.dct[item]

    def __contains__(self, item):
        """Determine if item in self.dct."""
        return item in self.dct

    def __iter__(self):
        """Iterate over keys."""
        return self.dct.__iter__()

    def __getattr__(self, key):
        """Get attribute."""
        return self.dct[key]

    def values(self):
        """Wrap self.dct.values."""
        return self.dct.values()

    def items(self):
        """Wrap self.dct.items."""
        return self.dct.items()

    def keys(self):
        """Wrap self.dct.keys."""
        return self.dct.keys()

    def webpanel(self):
        """Get web panel."""
        raise NotImplementedError


class GapResults(ASRResults):
    """Ground state results.

    Attributes
    ----------
    gap: float
        The band gap [eV].
    gap_dir: float
        The direct band gap [eV].
    """

    gap: float
    dipz: float


class GSResults(ASRResults):
    """Ground state results.

    Attributes
    ----------
    gap: float
        The band gap in eV.
    gaps_nosoc: GapResults
        Collection of band gap related results.
    """

    gap: float
    dipz: float
    gaps_nosoc: GapResults


print(GSResults)
print(GSResults.__dict__)
