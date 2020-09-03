"""Implements ASRResults object and related quantities."""
from typing import get_type_hints, List, Any
from ase.io import jsonio


def read_json(json):
    """Decode a json-serialized string to object."""
    return jsonio.decode(json)


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
        """Create ASE compatible webpanel."""
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
        panel = {'title': 'Basic electronic properties (PBE)',
                 'columns': columns,
                 'sort': 1}
        return [panel]


webpanel = WebPanel()


class UnknownASRResultsFormat(Exception):
    pass


protected_metadata_keys = {'version'}


class ASRResults:
    """Base class for describing results generated with recipes.

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
    prev_version: Any = None

    def __init__(self, metadata={}, **data):
        """Initialize results from dict."""
        self._data = data
        self.metadata = metadata

    @property
    def data(self) -> dict:
        """Get results data."""
        return self._data

    @property
    def metadata(self) -> dict:
        """Get results metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata) -> None:
        """Set results metadata."""
        assert not protected_metadata_keys.intersection(metadata.keys()), \
            f'You cannot write metadata with keys={protected_metadata_keys}.'
        metadata = {key: value for key, value in metadata.items()}
        metadata['version'] = self.version
        self._metadata = metadata

    @classmethod
    def from_data(cls, data, metadata, version):
        """Instantiate results from data, metadata, version specification."""
        # Walk through all previous implementations.
        while version != cls.version and cls.prev_version is not None:
            cls = cls.prev_version

        if not version == cls.version:
            raise UnknownASRResultsFormat(
                'Unknown version number: version={version}')
        return cls(**data, metadata=metadata)

    @classmethod
    def from_json(cls, json):
        """Initialize from json string."""
        tmp = read_json(json)
        metadata = tmp['metadata']
        version = metadata.pop('version')
        data = tmp['data']

        return cls.from_data(data, metadata, version)

    def __getitem__(self, item):
        """Get item from self.data."""
        return self.data[item]

    def __contains__(self, item):
        """Determine if item in self.data."""
        return item in self.data

    def __iter__(self):
        """Iterate over keys."""
        return self.data.__iter__()

    def __getattr__(self, key):
        """Get attribute."""
        if key in self.keys():
            return self.data[key]
        return super().__getattr__(self, key)

    def values(self):
        """Wrap self.data.values."""
        return self.data.values()

    def items(self):
        """Wrap self.data.items."""
        return self.data.items()

    def keys(self):
        """Wrap self.data.keys."""
        return self.data.keys()

    def get_formats(self):
        """Get implemented result formats."""
        formats = {'json': encode_json,
                   'html': encode_html,
                   'dict': encode_dict,
                   'ase_webpanel': webpanel}
        return formats

    def format_as(self, fmt: str = '') -> Any:
        """Format Results as string."""
        formats = self.get_formats()
        return formats[fmt](self)

    def __format__(self, fmt: str) -> str:
        """Encode results as string."""
        formats = self.get_formats()
        return formats[fmt](self)

    def __str__(self):
        """Convert data to string."""
        string_parts = []
        for key, value in self.items():
            string_parts.append(f'{key}={value}')
        return "\n".join(string_parts)

    def __eq__(self, other):
        """Compare two results objects."""
        if not isinstance(other, type(self)):
            return False
        return self.format_as('dict') == other.format_as('dict')


def encode_json(results: ASRResults):
    """Encode a ASRResults object as json."""
    from ase.io.jsonio import MyEncoder
    data = results.format_as('dict')
    return MyEncoder(indent=1).encode(data)


def encode_html(results: ASRResults):
    """Encode a ASRResults object as html."""
    return str(results)


def encode_dict(results: ASRResults):
    """Encode ASRResults object as dict."""
    return {'data': results.data,
            'metadata': results.metadata}
