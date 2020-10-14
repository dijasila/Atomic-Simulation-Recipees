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

"""
from ase.io import jsonio
import copy
import typing
from abc import ABC, abstractmethod
from . import get_recipe_from_name
import importlib


def read_old_data(dct) -> typing.Tuple[dict, dict, dict]:
    """Parse an old style result dictionary."""
    metadata = {}
    data = {}
    for key, value in dct.items():
        if key.startswith('__') and key.endswith('__'):
            key_name = key[2:-2]
            if key_name in MetaData.accepted_keys:
                metadata[key_name] = value
        else:
            data[key] = value
    asr_name = metadata['asr_name']
    recipe = get_recipe_from_name(asr_name)
    asr_obj_id = recipe.returns.get_obj_id()
    return data, metadata, 0, asr_obj_id


def read_new_data(dct) -> typing.Tuple[dict, dict, dict]:
    """Parse a new style result dictionary."""
    metadata = dct['metadata']
    data = dct['data']
    version = dct['version']
    asr_obj_id = dct['__asr_obj_id__']
    return data, metadata, version, asr_obj_id


class UnknownDataFormat(Exception):
    """Unknown ASR Result format."""

    pass


known_object_types = {'Result'}


def get_reader_function(dct):
    """Determine dataformat of dct and return approriate reader."""
    if '__asr_obj_id__' in dct:
        # Then this is a new-style data-format
        reader_function = read_new_data
    elif '__asr_name__' in dct:
        # Then this is a old-style data-format
        reader_function = read_old_data
    else:
        raise UnknownDataFormat
    return reader_function


def find_class_matching_version(returns, version):
    """Find result class that matches version.

    Walks through the class hierarchy defined by returns.prev_version and
    searches for class fulfilling returns.version == version.

    Raises :py:exc:`UnknownASRResultFormat` if no matching result is
    found.

    """
    # Walk through all previous implementations.
    while version != returns.version and returns.prev_version is not None:
        returns = returns.prev_version

    if not version == returns.version:
        raise UnknownASRResultFormat(
            'Unknown version number: version={version}')

    return returns


def get_class_matching_obj_id(asr_obj_id):
    assert asr_obj_id.startswith('asr.'), f'Invalid object id {asr_obj_id}'

    module, name = asr_obj_id.split('::')
    mod = importlib.import_module(module)
    cls = getattr(mod, name)

    return cls


def dct_to_result(dct):
    """Convert dict representing an ASR result to corresponding result object."""
    reader_function = get_reader_function(dct)
    data, metadata, version, asr_obj_id = reader_function(dct)

    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        try:
            data[key] = dct_to_result(value)
        except UnknownDataFormat:
            pass

    cls = get_class_matching_obj_id(asr_obj_id)
    cls = find_class_matching_version(cls, version)
    result = cls.fromdict(
        dict(data=data, metadata=metadata, version=version))
    return result


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


class JSONEncoder(ResultEncoder):
    """JSON ASRResult encoder."""

    def encode(self, result: 'ASRResult', indent=1):
        """Encode a ASRResult object as json."""
        from ase.io.jsonio import MyEncoder
        data = result.format_as('dict')
        return MyEncoder(indent=indent).encode(data)

    def decode(self, cls, json_string: str):
        """Decode json string."""
        dct = jsonio.decode(json_string)
        return cls.fromdict(dct)


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
        return result.todict()

    def decode(self, cls, dct: dict):
        """Decode dict."""
        return cls.fromdict(dct)


def get_object_descriptions(obj):
    """Get key descriptions of object."""
    return obj.key_descriptions


def get_object_types(obj):
    """Get type hints of object."""
    return typing.get_type_hints(obj)


def format_key_description_pair(key: str, attr_type: type, description: str):
    """Format a key-type-description for a docstring.

    Parameters
    ----------
    key
        Key name.
    attr_type
        Key type.
    description
        Documentation of key.
    """
    if attr_type.__module__ == 'builtins':
        type_desc = attr_type.__name__
    elif attr_type.__module__ == 'typing':
        type_desc = str(attr_type)
    else:
        type_desc = f'{attr_type.__module__}.{attr_type.__name__}'
    return (f'{key}: {type_desc}\n'
            f'    {description}')


def make_property(key, doc, return_type):

    def getter(self) -> return_type:
        return self.data[key]

    getter.__annotations__ = {'return': return_type}

    def setter(self, value) -> None:
        if self.data:
            raise AttributeError(
                f'Data was already set. You cannot overwrite/set data.'
            )
        self.data[key] = value

    return property(fget=getter, fset=setter, doc=doc)


def prepare_result(cls: object) -> str:
    """Prepare result class."""
    descriptions = get_object_descriptions(cls)
    types = get_object_types(cls)
    type_keys = set(types)
    description_keys = set(descriptions)
    missing_types = description_keys - type_keys
    assert not missing_types, f'{cls.get_obj_id()}: Missing types for={missing_types}.'

    data_keys = description_keys
    for key in descriptions:
        description = descriptions[key]
        attr_type = types[key]
        setattr(cls, key, make_property(key, description, return_type=attr_type))

    cls._strict = True
    cls._known_data_keys = data_keys
    return cls


class UnknownASRResultFormat(Exception):
    """Exception when encountering unknown results version number."""

    pass


class MetaDataNotSetError(Exception):
    """Error raised when encountering unknown metadata key."""

    pass


class MetaData:
    """Metadata object.

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
    >>> metadata.todict()
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
            assert key in self.accepted_keys, f'Unknown MetaData key={key}.'
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

    def todict(self):
        """Format metadata as dict."""
        return copy.deepcopy(self._dct)

    def __str__(self):
        """Represent as string."""
        dct = self.todict()
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
    overwrite the ``formats`` dictionary appropriately.

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
    >>> print(format(result, 'json'))
    {
     "data": {
      "a": 1
     },
     "metadata": {
      "resources": {
       "time": "a good time."
      }
     },
     "version": 0
    }
    >>> result.format_as('ase_webpanel', {}, {})
    [{'title': 'Results', \
'columns': [[{'type': 'table', 'header': ['key', 'value'], \
'rows': [['a', 1]]}]], 'sort': 1}]

    """ # noqa

    version: int = 0
    prev_version: typing.Any = None
    key_descriptions: typing.Dict[str, str]
    formats = {'json': JSONEncoder(),
               'html': HTMLEncoder(),
               'dict': DictEncoder(),
               'ase_webpanel': WebPanelEncoder(),
               'str': str}

    _strict = False

    def __init__(self, metadata={}, _strict=None, **data):
        """Initialize result from dict.

        Parameters
        ----------
        **data : key-value-pairs
            Input data to be wrapped.
        metadata : dict
            Dictionary containing metadata.
        """
        if (_strict is None and self._strict) or _strict:
            data_keys = set(data)
            unknown_keys = data_keys - self._known_data_keys
            assert not unknown_keys, \
                f'{self.get_obj_id()}: Trying to set unknown keys={unknown_keys}'
            missing_keys = self._known_data_keys - data_keys
            assert not missing_keys, \
                f'{self.get_obj_id()}: Missing data keys={missing_keys}'
        self._data = data
        self._metadata = MetaData()
        self.metadata.set(**metadata)

    @classmethod
    def get_obj_id(cls) -> str:
        return f'{cls.__module__}::{cls.__name__}'

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
        result = formats[format].decode(cls, input_data)
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

    def format_as(self, format: str = '', *args, **kwargs) -> typing.Any:
        """Format result in specific format."""
        formats = self.get_formats()
        return formats[format](self, *args, **kwargs)

    # To and from dict
    def todict(self):
        tmpdata = {}

        for key, value in self.data.items():
            if isinstance(value, ASRResult):
                value = value.todict()
            tmpdata[key] = value

        return {'__asr_obj_id__': self.get_obj_id(),
                'data': tmpdata,
                'metadata': self.metadata.todict(),
                'version': self.version}

    @classmethod
    def fromdict(cls, dct: dict):
        metadata = dct['metadata']
        version = dct['version']
        assert version == cls.version, \
            f'Inconsistent versions. data_version={version}, self.version={cls.version}'
        data = dct['data']
        result = cls(**data)
        result.metadata = metadata
        return result

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
        raise AttributeError

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
        print(self.get_obj_id(), repr(fmt))
        formats = self.get_formats()
        return formats[fmt](self)

    def __str__(self):
        """Convert data to string."""
        string_parts = []
        for key, value in self.items():
            string_parts.append(f'{key}=' + str(value))
        return "\n".join(string_parts)

    def __eq__(self, other):
        """Compare two result objects."""
        if not isinstance(other, type(self)):
            return False
        return self.data == other.data
