"""Implements ASRResult object and related quantities.

The most important class in this module is
:py:class:`asr.core.results.ASRResult`, which is used to wrap results
generated with ASR.

:py:class:`asr.core.results.ASRResult` has a bunch of associated
 encoders that implements different ways of representing results, and
 potentially also implements ways to decode results. These encoders are:

- :py:class:`asr.core.results.DictEncoder`
- :py:class:`asr.core.results.JSONEncoder`
- :py:class:`asr.core.results.HTMLEncoder`
- :py:class:`asr.core.results.WebPanelEncoder`

A dictionary representation of a result-object can be converted to a
result object through :py:func:`asr.core.results.dct_to_result`.

To support reading of different historical ASR dataformats we use an
intermediate :py:class:`DataContainer` before constructing the final
result-object.

"""
from ase.io import jsonio
import copy
from typing import get_type_hints, List, Any, Dict
from abc import ABC, abstractmethod
from . import get_recipe_from_name


def read_old_data(dct) -> 'DataContainer':
    """Parse an old style result dictionary."""
    metadata = {}
    data = {}
    for key, value in dct.items():
        if key.startswith('__') and key.endswith('__'):
            metadata[key[2:-2]] = value
        else:
            data[key] = value
    return DataContainer(data=data, metadata=metadata, version=0)


def read_new_data(dct) -> 'DataContainer':
    """Parse a new style result dictionary."""
    metadata = dct['metadata']
    data = dct['data']
    version = dct['version']
    return DataContainer(data=data, metadata=metadata, version=version)


class UnknownDataFormat(Exception):
    """Unknown ASR Result format."""

    pass


def dct_to_result(dct):
    """Parse dict and return result object."""
    if 'metadata' in dct:
        # Then this is a new-style data-format
        datacontainer = read_new_data(dct)
    elif '__asr_name__' in dct:
        # Then this is a old-style data-format
        datacontainer = read_old_data(dct)
    else:
        raise UnknownDataFormat
    metadata = datacontainer.get_metadata()
    asr_name = metadata['asr_name']
    recipe = get_recipe_from_name(asr_name)
    result = recipe.returns.from_format(datacontainer,
                                        format='datacontainer')
    return result


class DataContainer:
    """Abstract data format."""

    def __init__(self, data, metadata, version):
        dct = {'data': data,
               'metadata': metadata,
               'version': version}
        self._dct = dct

    def get_data(self):
        return self._dct['data']

    def get_version(self):
        return self._dct['version']

    def get_metadata(self):
        return self._dct['metadata']

    def set_metadata(self, metadata):
        self._dct['metadata'] = metadata


class ResultEncoder(ABC):
    """Abstract encoder base class.

    Encodes a results object as a specific format. Optionally
    provides functionality for decoding.

    """

    def __call__(self, result, *args, **kwargs):
        """Encode result."""
        return self.encode(result, *args, **kwargs)

    @abstractmethod
    def encode(self, result, *args, **kwargs):
        """Encode result."""
        raise NotImplementedError

    def decode(self, formatted_results):
        """Decode result."""
        return NotImplemented


class DataContainerEncoder(ResultEncoder):
    """DataContainer ASRResult encoder."""

    def encode(self, result: 'ASRResult'):
        """Encode a ASRResult object as a DataContainer."""
        dct = result.format_as('dict')
        return DataContainer(data=dct['data'],
                             metadata=dct['metadata'],
                             version=dct['version'])

    def decode(self, datacontainer: DataContainer):
        """Decode datacontainer."""
        metadata = datacontainer.get_metadata()
        data = datacontainer.get_data()
        version = datacontainer.get_version()
        return data, metadata, version


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
        data = tmp['data']
        version = tmp['version']
        return data, metadata, version


class HTMLEncoder(ResultEncoder):
    """HTML ASRResult encoder."""

    def encode(self, result: 'ASRResult'):
        """Encode a ASRResult object as html."""
        return str(result)


class WebPanelEncoder(ResultEncoder):
    """Encoder for ASE compatible webpanels."""

    def encode(self, result, row, key_descriptions):
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
        panel = {'title': 'Results',
                 'columns': columns,
                 'sort': 1}
        return [panel]


class DictEncoder(ResultEncoder):
    """Dict ASRResult encoder."""

    def encode(self, result: 'ASRResult'):
        """Encode ASRResult object as dict."""
        return {'data': result.data,
                'metadata': result.metadata.to_dict(),
                'version': result.version}

    def decode(self, dct: dict):
        """Decode decode dict to data, metadata, version."""
        metadata = dct['metadata']
        version = dct['version']
        data = dct['data']
        return data, metadata, version


def get_object_descriptions(obj):
    """Get key descriptions of object."""
    return obj.key_descriptions


def get_object_types(obj):
    """Get type hints of object."""
    return get_type_hints(obj)


def format_key_description_pair(key: str, attr_type: type, description: str):
    """Format a key-type-description for a docstring."""
    return (f'{key}: {attr_type}\n'
            f'    {description}')


def set_docstring(obj) -> str:
    """Parse key descriptions on object and make pretty docstring."""
    descriptions = get_object_descriptions(obj)
    types = get_object_types(obj)
    type_keys = set(types)
    description_keys = set(descriptions)
    assert set(descriptions).issubset(set(types)), description_keys - type_keys
    docstring_parts: List[str] = [obj.__doc__ or '',
                                  '',
                                  'Attributes',
                                  '----------']
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


class MetaDataNotSetError(Exception):
    """Error raised when encountering unknown metadata key."""

    pass


class MetaData:
    """Metadata object.

    Attributes
    ----------
    asr_name: str
        The name of the corresponding recipe.
    params: Dict[str, Any]
        The parameters associated with the results.
    resources: Dict[str, Union[str, float, int]]
        The resources used when computing result.
    code_versions: Dict[str, str]
        The code versions and possibly git hashes of associated code.
    requires: Dict[str, str]
        The required files and their MD5 hashes.
    creates: Dict[str, str]
        The created files nad their MD5 hashes.

    Methods
    -------
    __init__
        Construct MetaData object.
    __setattr__
        Set metadata attributes.
    set
        Set multiple metadata attributes.
    to_dict
        Represent metadata as dictionary.

    Examples
    --------
    >>> metadata = MetaData(asr_name='asr.gs')
    >>> metadata
    asr_name=asr.gs
    >>> metadata.code_versions = {'asr': '0.1.2'}
    >>> metadata
    asr_name=asr.gs
    code_versions={'asr': '0.1.2'}
    >>> metadata.set(resources={'time': 10}, params={'a': 1})
    >>> metadata
    asr_name=asr.gs
    code_versions={'asr': '0.1.2'}
    resources={'time': 10}
    params={'a': 1}
    >>> metadata.to_dict()
    {'asr_name': 'asr.gs', 'code_versions': {'asr': '0.1.2'},\
 'resources': {'time': 10}, 'params': {'a': 1}}
    """ # noqa

    accepted_keys = {'asr_name',
                     'params',
                     'resources',
                     'code_versions',
                     'creates',
                     'requires'}

    def __init__(self, **kwargs):
        """Initialize MetaData object."""
        self._dct = {}
        self.set(**kwargs)

    def set(self, **kwargs):
        """Set metadata values."""
        for key, value in kwargs.items():
            assert key in self.accepted_keys, 'Unknown MetaData key={key}.'
            setattr(self, key, value)

    def validate(self):
        """Assert integrity of metadata."""
        assert set(self._dct).issubset(self.accepted_keys)

    @property
    def asr_name(self):
        """For example 'asr.gs'."""
        return self._get('asr_name')

    @asr_name.setter
    def asr_name(self, value):
        """Set asr_name."""
        self._set('asr_name', value)

    @property
    def params(self):
        """Return dict containing parameters."""
        return self._get('params')

    @params.setter
    def params(self, value):
        """Set params."""
        self._set('params', value)

    @property
    def resources(self):
        """Return resources."""
        return self._get('resources')

    @resources.setter
    def resources(self, value):
        """Set resources."""
        self._set('resources', value)

    @property
    def code_versions(self):
        """Return code versions."""
        return self._get('code_versions')

    @code_versions.setter
    def code_versions(self, value):
        """Set code_versions."""
        self._set('code_versions', value)

    @property
    def creates(self):
        """Return list of created files."""
        return self._get('creates')

    @creates.setter
    def creates(self, value):
        """Set creates."""
        self._set('creates', value)

    @property
    def requires(self):
        """Return list of required files."""
        return self._get('requires')

    @requires.setter
    def requires(self, value):
        """Set requires."""
        self._set('requires', value)

    def _set(self, key, value):
        self._dct[key] = value

    def _get(self, key):
        if key not in self._dct:
            raise MetaDataNotSetError(f'Metadata key={key} has not been set!')
        return self._dct[key]

    def to_dict(self):
        """Format metadata as dict."""
        return copy.deepcopy(self._dct)

    def __str__(self):
        """Represent as string."""
        dct = self.to_dict()
        lst = []
        for key, value in dct.items():
            lst.append(f'{key}={value}')
        return '\n'.join(lst)

    def __repr__(self):
        """Represent object."""
        return str(self)


class ASRResult(object):
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
    >>> result.metadata = {'resources': {'time': 'a good time.'}}
    >>> result.a
    1
    >>> result['a']
    1
    >>> result.metadata
    resources={'time': 'a good time.'}
    >>> str(result)
    'a=1'
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
               'ase_webpanel': WebPanelEncoder(),
               'datacontainer': DataContainerEncoder()}

    def __init__(self, metadata={}, **data):
        """Initialize result from dict.

        Parameters
        ----------
        **data : key-value-pairs
            Input data to be wrapped.
        metadata : dict
            Dictionary containing metadata.
        """
        self._data = data
        self._metadata = MetaData()
        self.metadata.set(**metadata)

    @property
    def data(self) -> dict:
        """Get result data."""
        return self._data

    @property
    def metadata(self) -> MetaData:
        """Get result metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, metadata) -> None:
        """Set result metadata."""
        # The following line is for copying the metadata into place.
        metadata = copy.deepcopy(metadata)
        self._metadata.set(**metadata)

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
        result = cls(**data)
        result.metadata = metadata
        return result

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

    def format_as(self, format: str = '', *args, **kwargs) -> Any:
        """Format result in specific format."""
        formats = self.get_formats()
        return formats[format](self, *args, **kwargs)

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

    def get(self, key, *args):
        """Wrap self.data.get."""
        return self.data.get(key, *args)

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
        return self.data == other.data
