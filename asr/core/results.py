"""Implements ASRResults object and related quantities."""
from typing import get_type_hints, List, Any


def get_object_descriptions(obj):
    """Get key descriptions of object."""
    return obj.key_descriptions


def get_object_types(obj):
    """Get type hints of object."""
    return get_type_hints(obj)


def format_key_description_pair(key: str, attr_type: type, description: str):
    """Format a key-type-description for a docstring."""
    return f'{key}: {attr_type}\n    {description}'


def set_docstring(obj) -> str:
    """Parse key descriptions on object and make pretty docstring."""
    descriptions = get_object_descriptions(obj)
    types = get_object_types(obj)
    type_keys = set(types)
    description_keys = set(descriptions)
    assert set(descriptions).issubset(set(types)), description_keys - type_keys
    docstring_parts: List[str] = [obj.__doc__ or '',
                                  '', 'Parameters', '----------']
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
    prev_version : ASRResults or None
        Pointer to a previous results format. If none, this is the
        final version.
    """

    version: int = 0
    webpanel: WebPanel = WebPanel()
    prev_version: Any = None

    def __init__(self, metadata={}, **dct):
        """Initialize results from dict."""
        self._dct = dct
        self.metadata = metadata

    def to_json(self, filename):
        """Write results to file."""
        from asr.core import write_json
        write_json(filename,
                   dict(data=self._dct,
                        metadata=self.metadata))

    @classmethod
    def from_json(cls):
        from asr.core import read_json
        tmp = read_json(filename)
        return cls(**tmp['data'], metadata=tmp['metadata'])

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

    def set_metadata(self, metadata):
        """Set results metadata."""
        self.metadata = metadata

    def get_metadata(self):
        """Get results metadata."""
        return self.metadata

    def __format__(self, fmt: str = '') -> Any:
        """Format Results as string."""
        formats = {'json': self.encode_json,
                   'html': self.encode_html,
                   'ase_webpanel': self.webpanel}

        return formats[fmt]()

    def __str__(self):
        """Convert data to string."""
        string_parts = []
        for key, value in self.items():
            string_parts.append(f'{key}={value}')
        return "\n".join(string_parts)
