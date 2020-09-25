"""Implements ASRResult object and related quantities."""
from ase.io import jsonio
import copy
from typing import get_type_hints, List, Any, Dict
from abc import ABC, abstractmethod


class ResultEncoder(ABC):
    """Abstract encoder base class.

    Encodes a results object as a specific format. Optionally
    provides functionality for decoding.

    """

    def __call__(self, result):
        """Encode result."""
        return self.encode(result)

    @abstractmethod
    def encode(self, result):
        """Encode result."""
        raise NotImplementedError

    def decode(self, formatted_results):
        """Decode result."""
        return NotImplemented


class JSONEncoder(ResultEncoder):
    """JSON ASRResult encoder."""

    def encode(self, result: 'ASRResult'):
        """Encode a ASRResult object as json."""
        from ase.io.jsonio import MyEncoder
        data = result.format_as('dict')
        return MyEncoder(indent=1).encode(data)

    def decode(self, json_string: str):
        """Decode json string."""
        tmp = jsonio.decode(json_string)
        metadata = tmp['metadata']
        version = metadata.pop('version')
        data = tmp['data']
        return data, metadata, version


class HTMLEncoder(ResultEncoder):
    """HTML ASRResult encoder."""

    def encode(self, result: 'ASRResult'):
        """Encode a ASRResult object as html."""
        return str(result)


class WebPanelEncoder(ResultEncoder):
    """Encoder for ASE compatible webpanels."""

    def encode(self, result):
        """Make basic webpanel.

        Simply prints all attributes.
        """
        rows = []
        for key, value in result.items():
            rows.append([key, value])
        table = {'type': 'table',
                 'header': ['key', 'value'],
                 'rows': rows}
        columns = [[table]]
        panel = {'title': 'Basic electronic properties (PBE)',
                 'columns': columns,
                 'sort': 1}
        return [panel]


class DictEncoder(ResultEncoder):
    """Dict ASRResult encoder."""

    def encode(self, result: 'ASRResult'):
        """Encode ASRResult object as dict."""
        return {'data': result.data,
                'metadata': result.metadata}


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
                                  '',
                                  'Data attributes',
                                  '---------------']
    for key in descriptions:
        description = descriptions[key]
        attr_type = types[key]
        string = format_key_description_pair(key, attr_type, description)
        docstring_parts.append(string)
    docstring = "\n".join(docstring_parts)

    obj.__doc__ = docstring
    return obj


class UnknownASRResultFormat(Exception):
    """Exception when encountering unknown results version number."""

    pass


protected_metadata_keys = {'version'}


class ASRResult:
    """Base class for describing results generated with recipes.

    A results object is a container for results generated with ASR. It
    contains data and metadata describing results and the
    circumstances under which the result were generated,
    respectively. The metadata has to be set manually through the
    ``metadata`` property. The wrapped data can be accessed through
    the ``data`` property or directly as an attribute on the object.

    The result object provides the means of presenting the wrapped
    data in different formats as obtained from the ``get_formats``
    method. To implement a new webpanel, inherit from this class and
    overwrite the ``get_formats`` method appropriately.

    This object implements dict/namespace like default behaviour and
    contained data can be check with ``in`` (see "Examples" below).

    Examples
    --------
    >>> result = ASRResult(a=1)
    >>> result.metadata = {'time': 'a good time.'}
    >>> result.a
    1
    >>> result['a']
    1
    >>> result.metadata
    {'time': 'a good time.'}
    >>> str(result)
    a=1
    >>> 'a' in result
    True
    >>> other_result = ASRResult(a=1)
    >>> result == other_result
    True

    Attributes
    ----------
    data
        Associated result data.
    metadata
        Associated result metadata.
    version : int
        The version number.
    prev_version : ASRResult or None
        Pointer to a previous result format. If none, this is the
        final version.
    key_descriptions: Dict[str, str]
        Description of data attributes

    Methods
    -------
    get_formats
        Return implemented formats.
    from_format
        Decode and instantiate result object from format.
    format_as
        Encode result in a specific format.

    """

    version: int = 0
    prev_version: Any = None
    key_descriptions: Dict[str, str]
    formats = {'json': JSONEncoder(),
               'html': HTMLEncoder(),
               'dict': DictEncoder(),
               'ase_webpanel': WebPanelEncoder()}

    def __init__(self, metadata={}, **data):
        """Initialize result from dict.

        Parameters
        ----------
        **data : key-value-pairs
            Input data to be wrapped.
        metadata : dict
            Extra metadata describing code versions etc.
        """
        self._data = data
        self.metadata = metadata

        kd_keys = self.key_descriptions.keys()
        data_keys = data.keys()
        assert set(kd_keys) == set(data_keys), \
            (kd_keys, data_keys)

    @property
    def data(self) -> dict:
        """Get result data."""
        return self._data

    @property
    def metadata(self) -> dict:
        """Get result metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata) -> None:
        """Set result metadata."""
        assert not protected_metadata_keys.intersection(metadata.keys()), \
            f'You cannot write metadata with keys={protected_metadata_keys}.'
        # The following line is for copying the metadata into place.
        metadata = copy.deepcopy(metadata)
        metadata['version'] = self.version
        self._metadata = metadata

    @classmethod
    def from_format(cls, input_data, format='json'):
        """Instantiate result from format."""
        formats = cls.get_formats()
        data, metadata, version = formats[format].decode(input_data)
        # Walk through all previous implementations.
        while version != cls.version and cls.prev_version is not None:
            cls = cls.prev_version

        if not version == cls.version:
            raise UnknownASRResultFormat(
                'Unknown version number: version={version}')
        return cls(**data, metadata=metadata)

    @classmethod
    def get_formats(cls) -> dict:
        """Get implemented result formats."""
        try:
            sup_formats = super(cls, cls).formats
            my_formats = {key: value for key, value in sup_formats.items()}
        except AttributeError:
            my_formats = {}
        my_formats.update(cls.formats)
        return my_formats

    def format_as(self, fmt: str = '', **kwargs) -> Any:
        """Format Result as string."""
        formats = self.get_formats()
        return formats[fmt](self, **kwargs)

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
        """Encode result as string."""
        formats = self.get_formats()
        return formats[fmt](self)

    def __str__(self):
        """Convert data to string."""
        string_parts = []
        for key, value in self.items():
            string_parts.append(f'{key}={value}')
        return "\n".join(string_parts)

    def __eq__(self, other):
        """Compare two result objects."""
        if not isinstance(other, type(self)):
            return False
        return self.format_as('dict') == other.format_as('dict')
