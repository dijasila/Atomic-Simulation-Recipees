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
result object through :py:func:`asr.core.results.decode_object`.

"""
import copy
import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping
from .utils import get_recipe_from_name
import importlib
import inspect
import warnings


def read_hacked_data(dct) -> 'ObjectDescription':
    """Fix hacked results files to contain necessary metadata."""
    from .resultfile import fix_recipe_name_if_recipe_has_been_moved
    data = {}
    metadata = {}
    for key, value in dct.items():
        if key.startswith('__') and key.endswith('__'):
            key_name = key[2:-2]
            if key_name in MetaData.accepted_keys:
                metadata[key_name] = value
        else:
            data[key] = value
    asr_name = dct['__asr_hacked__']
    asr_name = fix_recipe_name_if_recipe_has_been_moved(asr_name)
    metadata['asr_name'] = asr_name
    recipe = get_recipe_from_name(asr_name)
    if issubclass(recipe.returns, ASRResult):
        returns = recipe.returns
    else:
        returns = ASRResult
    object_id = obj_to_id(returns)
    obj_desc = ObjectDescription(
        object_id=object_id,
        args=(),
        kwargs={'data': data,
                'metadata': metadata,
                'strict': False},
    )
    return obj_desc


def read_old_data(dct) -> 'ObjectDescription':
    """Parse an old style result dictionary."""
    from .resultfile import fix_recipe_name_if_recipe_has_been_moved
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
    asr_name = fix_recipe_name_if_recipe_has_been_moved(asr_name)
    asr_name = asr_name.replace("@", ":")
    metadata['asr_name'] = asr_name
    recipe = get_recipe_from_name(asr_name)
    if issubclass(recipe.returns, ASRResult):
        returns = recipe.returns
    else:
        returns = ASRResult
    object_description = ObjectDescription(
        object_id=obj_to_id(returns),
        args=(),
        kwargs=dict(
            data=data,
            metadata=metadata,
            strict=False
        ),
    )
    return object_description


def read_new_data(dct) -> 'ObjectDescription':
    """Parse a new style result dictionary."""
    object_description = ObjectDescription.fromdict(dct)
    return object_description


class UnknownDataFormat(Exception):
    """Unknown ASR Result format."""

    pass


known_object_types = {'Result'}


def get_reader_function(dct):
    """Determine dataformat of dct and return approriate reader."""
    if 'object_id' in dct:
        # Then this is a new-style data-format
        reader_function = read_new_data
    elif '__asr_name__' in dct:
        # Then this is a old-style data-format
        reader_function = read_old_data
    elif '__asr_hacked__' in dct:
        reader_function = read_hacked_data
    else:
        raise UnknownDataFormat(f"""

        Error when reading results file. The file contains the
        following data keys

            data_keys={dct.keys()}

        from which the data format could not be deduced.  If you
        suspect the reason is that the data is very old, it is
        possible that this could be fixed by running:

            $ python -m asr.utils.fix_object_ids folder1/ folder2/ ...

        where folder1 and folder2 are folders containing 'problematic'
        result files. If you have multiple folders that contains
        problematic files you can similarly to something like:

            $ python -m asr.utils.fix_object_ids */

        """)
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


class ModuleNameIsCorrupt(Exception):

    pass


def get_object_matching_obj_id(asr_obj_id):
    from .resultfile import fix_recipe_name_if_recipe_has_been_moved
    try:
        module, name = asr_obj_id.split('::')
    except ValueError:
        module, name = asr_obj_id.split(':')

    module = fix_recipe_name_if_recipe_has_been_moved(module)

    if module in {'None.None', '__main__'}:
        raise ModuleNameIsCorrupt(
            """
            There is a problem with your result objectid module name={module}. '
            This is a known bug. To fix the faulty result '
            files please run: '
            "python -m asr.utils.fix_object_ids folder1/ '
            folder2/ ..." '
            where folder1 and folder2 are folders containing '
            problematic result files.'"""
        )

    assert asr_obj_id.startswith(
        ('asr.', 'ase.', 'htwutil.')), f'Invalid object id {asr_obj_id}'
    mod = importlib.import_module(module)
    cls = getattr(mod, name)

    assert cls
    return cls


def object_description_to_object(object_description: 'ObjectDescription'):
    """Instantiate object description."""
    return object_description.instantiate()


def dct_to_result(dct: dict) -> typing.Any:
    """Convert dict representing an ASR result to corresponding result object."""
    warnings.warn(
        """

        'asr.core.dct_to_result' will change name to
        'asr.core.decode_object' in the future. Please update your
        scripts to reflect this change.""",
        DeprecationWarning,
    )

    return decode_object(dct)


def encode_object(obj: typing.Any):
    """Encode object such that it can be deserialized with `decode_object`."""
    from ase.io import jsonio
    if isinstance(obj, dict):
        newobj = {}
        for key, value in obj.items():
            newobj[key] = encode_object(value)
    elif isinstance(obj, list):
        newobj = []
        for value in obj:
            newobj.append(encode_object(value))
    elif isinstance(obj, tuple):
        newobj = tuple(encode_object(value) for value in obj)
    elif hasattr(obj, 'todict'):
        newobj = encode_object(jsonio.MyEncoder().default(obj))
    elif hasattr(obj, 'to_object_desc'):
        newobj = encode_object(obj.to_object_desc())
    else:
        newobj = obj
    return newobj


def decode_object(obj: typing.Any) -> typing.Any:
    """Convert object representing an ASR result to corresponding result object."""
    from ase.io import jsonio
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = decode_object(value)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = decode_object(value)
    elif isinstance(obj, tuple):
        obj = tuple(decode_object(value) for value in obj)

    if isinstance(obj, dict):
        obj = jsonio.object_hook(obj)

    if isinstance(obj, dict):
        try:
            obj = decode_result(obj)
        except UnknownDataFormat:
            pass

    return obj


def decode_result(dct: dict) -> 'ASRResult':
    reader_function = get_reader_function(dct)
    object_description = reader_function(dct)
    obj = object_description_to_object(object_description)
    return obj


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
        from ase.io import jsonio
        dct = jsonio.decode(json_string)
        return cls.from_object_desc(dct)


class HTMLEncoder(ResultEncoder):
    """HTML ASRResult encoder."""

    def encode(self, result: 'ASRResult'):
        """Encode a ASRResult object as html."""
        return str(result)


class WebPanelEncoder(ResultEncoder):
    """Encoder for ASE compatible webpanels."""

    def encode(self, result, context):
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
        return result.to_object_desc()

    def decode(self, cls, dct: dict):
        """Decode dict."""
        return cls.from_object_desc(dct)


def get_key_descriptions(obj):
    """Get key descriptions of object."""
    if hasattr(obj, 'key_descriptions'):
        return obj.key_descriptions
    return {}


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
                'Data was already set. You cannot overwrite/set data.')
        self.data[key] = value

    return property(fget=getter, fset=setter, doc=doc)


def prepare_result(cls: object) -> str:
    """Prepare result class.

    This function read key descriptions and types defined in a Result class and
    assigns properties to all keys. It also sets strict=True used by the
    result object to ensure all data is present. It also changes the signature
    of the class to something more helpful than args, kwargs.

    """
    descriptions = get_key_descriptions(cls)
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

    sig = inspect.signature(cls.__init__)
    parameters = [list(sig.parameters.values())[0]] + [
        inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for key in data_keys
    ]
    sig = sig.replace(parameters=parameters)

    def __init__(self, *args, **kwargs):
        return super(type(self), self).__init__(*args, **kwargs)

    cls.__init__ = __init__
    cls.__init__.__signature__ = sig
    cls.strict = False
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
    >>> metadata = MetaData(asr_name='asr.c2db.gs')
    >>> metadata
    Metadata(asr_name=asr.c2db.gs)
    >>> metadata.code_versions = {'asr': '0.1.2'}
    >>> metadata
    Metadata(asr_name=asr.c2db.gs,code_versions={'asr': '0.1.2'})
    >>> metadata.set(resources={'time': 10}, params={'a': 1})
    >>> metadata
    Metadata(asr_name=asr.c2db.gs,code_versions={'asr': '0.1.2'},resources={'time': 10},params={'a': 1})
    >>> metadata.todict()
    {'asr_name': 'asr.c2db.gs', 'code_versions': {'asr': '0.1.2'},\
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
        """For example 'asr.c2db.gs'."""
        # Can this attribute be initialized from outdated files?
        # I don't know, but let's defend against that:
        name = self._get('asr_name')
        name = name.replace('@', ':')
        if name.endswith(':main'):
            name = name.rsplit(':', 1)[0]
        return name

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

    def __repr__(self):
        """Represent as string."""
        dct = self.todict()
        lst = []
        for key, value in dct.items():
            lst.append(f'{key}={value}')
        return 'Metadata(' + ','.join(lst) + ')'

    def __contains__(self, key):
        """Is metadata key set."""
        return key in self._dct


def obj_to_id(cls):
    """Get a string representation of path to object.

    Ie. if obj is the ASRResult class living in the module, asr.core.results,
    the correspinding string would be 'asr.core.results:ASRResult'.

    """
    module = inspect.getmodule(cls)
    path = module.__file__
    package = module.__package__
    assert package is not None, \
        ('Something went wrong in package identification.'
         'Please contact developer.')
    modulename = inspect.getmodulename(path)
    objname = cls.__name__

    assert modulename != '__main__', \
        ('Something went wrong in module name identification. '
         'Please contact developer.')

    return f'{package}.{modulename}:{objname}'


class ObjectDescription:
    """Result object descriptor."""

    def __init__(self, object_id: str, args: tuple, kwargs: dict,
                 constructor: typing.Optional[str] = None):
        """Initialize instance.

        Parameters
        ----------
        object_id: str
            ID of object, eg. 'asr.core.results:ASRResult' as
            produced by :py:func:`obj_to_id`.
        args
            Arguments for object construction.
        kwargs
            Keyword arguments for object construction.
        constructor: str or None
            ID of constructor object, ie. callable that can be used to
            instantiate object. If unset use constructor=object_id.
        """
        self._data = {
            'object_id': copy.copy(object_id),
            'constructor': (copy.copy(constructor) if constructor
                            else copy.copy(object_id)),
            'args': copy.deepcopy(args),
            'kwargs': copy.deepcopy(kwargs),
        }

    @property
    def object_id(self):
        """Get object id."""
        return self._data['object_id']

    @property
    def constructor(self):
        """Get object id."""
        return self._data['constructor']

    @property
    def args(self):
        """Get extra arguments supplied to constructor."""
        return self._data['args']

    @property
    def kwargs(self):
        """Get extra arguments supplied to constructor."""
        return self._data['kwargs']

    def todict(self):
        """Convert object description to dictionary."""
        return copy.deepcopy(self._data)

    @classmethod
    def fromdict(cls, dct):
        """Instantiate ObjectDescription from dict."""
        return cls(**dct)

    def instantiate(self):
        """Instantiate object."""
        cls = get_object_matching_obj_id(self.constructor)
        return cls(*self.args, **self.kwargs)


def data_to_dict(dct):
    """Recursively .todict all result instances."""
    for key, value in dct.items():
        if isinstance(value, ASRResult):
            dct[key] = value.todict()
            data_to_dict(dct[key])
    return dct


class ASRResult(Mapping):
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
    >>> @prepare_result
    ... class Result(ASRResult):
    ...     a: int
    ...     key_descriptions = {'a': 'Some key description.'}
    >>> result = Result.fromdata(a=1)
    >>> result.metadata = {'resources': {'time': 'a good time.'}}
    >>> result.a
    1
    >>> result['a']
    1
    >>> result.metadata
    Metadata(resources={'time': 'a good time.'})
    >>> str(result)
    'Result(a=1)'
    >>> 'a' in result
    True
    >>> other_result = Result.fromdata(a=1)
    >>> result == other_result
    True
    >>> print(format(result, 'json'))
    {
     "object_id": "asr.core.results:Result",
     "constructor": "asr.core.results:Result",
     "args": [],
     "kwargs": {
      "data": {
       "a": 1
      },
      "metadata": {
       "resources": {
        "time": "a good time."
       }
      },
      "strict": null
     }
    }
    """ # noqa

    version: int = 0
    prev_version: typing.Any = None
    key_descriptions: typing.Dict[str, str]
    formats = {'json': JSONEncoder(),
               'html': HTMLEncoder(),
               'dict': DictEncoder(),
               'str': str}

    strict = False
    _known_data_keys = set()

    def __init__(self,
                 data: typing.Dict[str, typing.Any] = None,
                 metadata: typing.Dict[str, typing.Any] = None,
                 strict: typing.Optional[bool] = None):
        """Instantiate result.

        Parameters
        ----------
        data: Dict[str, Any]
            Input data to be wrapped.
        metadata: dict
            Dictionary containing metadata.
        strict: bool or None
            Strictly enforce data entries in data.

        """
        strict = ((strict is None and self.strict) or strict)
        self.strict = strict
        if data is None:
            data = {}
        self._data = data

        if metadata is None:
            metadata = {}

        self._metadata = MetaData()
        self.metadata.set(**metadata)

        missing_keys = self.get_missing_keys()
        unknown_keys = self.get_unknown_keys()
        msg_ukwn = f'{self.get_obj_id()}: Trying to set unknown keys={unknown_keys}'
        msg_miss = f'{self.get_obj_id()}: Missing data keys={missing_keys}'
        if False:
            assert not missing_keys, msg_miss
            assert not unknown_keys, msg_ukwn

    def get_missing_keys(self):
        data_keys = set(self)
        missing_keys = self._known_data_keys - data_keys
        return missing_keys

    def get_unknown_keys(self):
        data_keys = set(self)
        unknown_keys = data_keys - self._known_data_keys
        return unknown_keys

    @classmethod
    def fromdata(cls, **data):
        return cls(data=data)

    @classmethod
    def get_obj_id(cls) -> str:
        return obj_to_id(cls)

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

    def get_object_desc(self) -> ObjectDescription:
        """Make ObjectDescription of this instance."""
        return ObjectDescription(
            object_id=obj_to_id(type(self)),
            args=(),
            kwargs={
                'data': self.data,
                'metadata': self.metadata,
                'strict': self.strict,
                # 'version': self.version,
            },
        )

    # To and from dict
    def to_object_desc(self):
        object_description = self.get_object_desc()
        return encode_object(object_description)

    @classmethod
    def from_object_desc(cls, dct: dict):
        obj_desc = ObjectDescription.fromdict(dct)
        return obj_desc.instantiate()

    # ---- Magic methods ----

    def __getitem__(self, item):
        """Get item from self.data."""
        return self._data[item]

    def __iter__(self):
        """Iterate over keys."""
        return iter(self._data)

    def __format__(self, fmt: str) -> str:
        """Encode result as string."""
        if fmt == '':
            fmt = 'str'
        formats = self.get_formats()
        return formats[fmt](self)

    def __str__(self):
        """Convert data to string."""
        string_parts = []
        for key, value in self.items():
            string_parts.append(f'{key}=' + str(value))
        return 'Result(' + ",".join(string_parts) + ')'

    def __len__(self):
        return len(self.data)


class HackedASRResult(ASRResult):
    pass
