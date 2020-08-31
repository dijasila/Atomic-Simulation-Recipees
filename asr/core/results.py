"""Implements ASRResults object and related quantities."""
from typing import get_type_hints, List, Any


def get_object_descriptions(obj):
    return obj.key_descriptions


def get_object_types(obj):
    return get_type_hints(obj)


def format_key_description_pair(key: str, attr_type: type, description: str):
    return f'{key}: {attr_type}\n    {description}'


def set_docstring(obj) -> str:
    descriptions = get_object_descriptions(obj)
    types = get_object_types(obj)
    type_keys = set(types)
    description_keys = set(descriptions)
    assert set(descriptions).issubset(set(types)), (type_keys, description_keys)
    docstring_parts: List[str] = [obj.__doc__ or '', '', 'Parameters', '----------']
    for key in descriptions:
        description = descriptions[key]
        attr_type = types[key]
        string = format_key_description_pair(key, attr_type, description)
        docstring_parts.append(string)
    docstring = "\n".join(docstring_parts)

    obj.__doc__ = docstring
    return obj


class WebPanel:
    """Web-panel for presenting results."""

    def __call__(self, results):
        return self.webpanel(results)

    def webpanel(self, results):
        """Make basic webpanel.

        Simply prints all attributes.
        """
        rows = []
        for key, value in results.items():
            rows.append([key, value])
        table = {'type': 'table',
                 'header': ['key', 'value'],
                 'rows': rows}
        columns = [[table]]
        return columns


class ASRResults:
    """Base class for describing results generated with recipes.

    WIP: Over time, by default, this class should be immutable.

    Attributes
    ----------
    version : int
        The version number.
    webpanel : WebPanel
        Functionality for producting ASE compatible web-panels.
    prev_version : ASRResults
        Pointer to a previous results format.
    """

    version: int = 0
    webpanel: WebPanel = WebPanel()
    prev_version: Any = None

    def __init__(self, **dct):
        """Initialize results from dict."""
        self._dct = dct

    def __getitem__(self, item):
        """Get item from self.dct."""
        return self._dct[item]

    def __contains__(self, item):
        """Determine if item in self.dct."""
        return item in self._dct

    def __iter__(self):
        """Iterate over keys."""
        return self._dct.__iter__()

    def __getattr__(self, key):
        """Get attribute."""
        if key in self.keys():
            return self._dct[key]
        return self.key

    def values(self):
        """Wrap self._dct.values."""
        return self._dct.values()

    def items(self):
        """Wrap self._dct.items."""
        return self._dct.items()

    def keys(self):
        """Wrap self._dct.keys."""
        return self._dct.keys()

    def __str__(self):
        """Convert data to string."""
        string_parts = []
        for key, value in self.items():
            string_parts.append(f'{key}={value}')
        return "\n".join(string_parts)
