"""Implements ASRResults object and related quantities."""
from ase.io import jsonio
import copy
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
    """Exception when encountering unknown results version number."""

    pass


protected_metadata_keys = {'version'}


class ASRResults:
    """Base class for describing results generated with recipes.

    A results object is a container for results generated with ASR. It
    contains data and metadata describing results and the
    circumstances under which the results were generated,
    respectively. The metadata has to be set manually through the
    ``metadata`` property. The wrapped data can be accessed through
    the ``data`` property or directly as an attribute on the object.

    The results object provides the means of presenting the wrapped
    data in different formats as obtained from the ``get_formats``
    method. To implement a new webpanel, inherit from this class and
    overwrite the ``get_formats`` method appropriately.

    This object implements dict/namespace like default behaviour and
    contained data can be check with ``in`` (see "Examples" below).

    Examples
    --------
    >>> results = ASRResults(a=1)
    >>> results.metadata = {'time': 'a good time.'}
    >>> results.a
    1
    >>> results['a']
    1
    >>> results.metadata
    {'time': 'a good time.'}
    >>> str(results)
    a=1
    >>> 'a' in results
    True
    >>> other_results = ASRResults(a=1)
    >>> results == other_results
    True

    Attributes
    ----------
    data
        Associated results data.
    metadata
        Associated results metadata.
    version : int
        The version number.
    prev_version : ASRResults or None
        Pointer to a previous results format. If none, this is the
        final version.

    Methods
    -------
    get_formats
        Return implemented formats.
    from_format
        Decode and instantiate results object from format.
    format_as
        Encode results in a specific format.

    """

    version: int = 0
    prev_version: Any = None

    def __init__(self, metadata={}, **data):
        """Initialize results from dict.

        Parameters
        ----------
        **data : key-value-pairs
            Input data to be wrapped.
        metadata : dict
            Extra metadata describing code versions etc.
        """
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
        # The following line is for copying the metadata into place.
        metadata = copy.deepcopy(metadata)
        metadata['version'] = self.version
        self._metadata = metadata

    @classmethod
    def from_format(cls, input_data, format='json'):
        """Instantiate results from format."""
        formats = cls.get_formats()
        data, metadata, version = formats[format]['decode'](input_data)
        # Walk through all previous implementations.
        while version != cls.version and cls.prev_version is not None:
            cls = cls.prev_version

        if not version == cls.version:
            raise UnknownASRResultsFormat(
                'Unknown version number: version={version}')
        return cls(**data, metadata=metadata)

    @staticmethod
    def get_formats() -> dict:
        """Get implemented result formats."""
        formats = {'json': {'encode': encode_json, 'decode': decode_json},
                   'html': {'encode': encode_html},
                   'dict': {'encode': encode_dict},
                   'ase_webpanel': {'encode': webpanel}}
        return formats

    def format_as(self, fmt: str = '') -> Any:
        """Format Results as string."""
        formats = self.get_formats()
        return formats[fmt]['encode'](self)

    # ---- Magic methods ----

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

    def __format__(self, fmt: str) -> str:
        """Encode results as string."""
        formats = self.get_formats()
        return formats[fmt]['encode'](self)

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


def decode_json(json_string: str):
    """Decode json string."""
    tmp = jsonio.decode(json_string)
    metadata = tmp['metadata']
    version = metadata.pop('version')
    data = tmp['data']
    return data, metadata, version


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
