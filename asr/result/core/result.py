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
import importlib
import inspect
import warnings

from collections.abc import Mapping

from ase.io import jsonio
from asr.result.core.exceptions import (UnknownDataFormat, UnknownASRResultFormat,
                                        ModuleNameIsCorrupt)
from asr.result.core.encoders import JSONEncoder, HTMLEncoder, DictEncoder
from asr.result.core.metadata import MetaData


known_object_types = {'Result'}


def parse_mod_func(name):
    # Split a module function reference like
    # asr.relax@main into asr.relax and main.
    mod, *func = name.split('@')
    if not func:
        func = ['main']

    assert len(func) == 1, \
        'You cannot have multiple : in your function description'

    return mod, func[0]


def get_recipe_from_name(name):
    # Get a recipe from a name like asr.gs@postprocessing
    import importlib
    assert name.startswith('asr.'), \
        'Not allowed to load recipe from outside of ASR.'
    mod, func = parse_mod_func(name)
    module = importlib.import_module(mod)
    return getattr(module, func)


def read_hacked_data(dct) -> 'ObjectDescription':
    """Fix hacked results files to contain necessary metadata."""
    data = {}
    metadata = {}
    for key, value in dct.items():
        if key.startswith('__') and key.endswith('__'):
            key_name = key[2:-2]
            if key_name in MetaData.accepted_keys:
                metadata[key_name] = value
        else:
            data[key] = value
    recipe = get_recipe_from_name(dct['__asr_hacked__'])
    object_id = obj_to_id(recipe.returns)
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
    object_description = ObjectDescription(
        object_id=obj_to_id(recipe.returns),
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


def get_reader_function(dct):
    """Determine dataformat of dct and return the appropriate reader."""
    if 'object_id' in dct:
        # Then this is a new-style data-format
        reader_function = read_new_data
    elif '__asr_name__' in dct:
        # Then this is an old-style data-format
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
    """Find result class that matches the version.

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


def obj_to_id(cls):
    """Get a string representation of the path to object.

    I.e., if obj is the ASRResult class living in the module, asr.core.results,
    the corresponding string would be 'asr.core.results::ASRResult.'

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

    return f'{package}.{modulename}::{objname}'


def get_object_matching_obj_id(asr_obj_id):
    module, name = asr_obj_id.split('::')
    if module in {'None.None', '__main__'}:
        raise ModuleNameIsCorrupt(
            """
            There is a problem with your result objectid module name={module}.'
            This is a known bug. To fix the faulty result '
            files please run: '
            "python -m asr.utils.fix_object_ids folder1/ '
            folder2/ ..." '
            where folder1 and folder2 are folders containing '
            problematic result files.'"""
        )

    assert asr_obj_id.startswith('asr.'), f'Invalid object id {asr_obj_id}'
    try:
        # Import the module dynamically
        mod = importlib.import_module(module)
        # Get the class or function from the module
        result_handler = getattr(mod, name)
        return result_handler
    except AttributeError:
        moved_module = "asr.paneldata"
        panel_names = {'asr.polarizability': 'OpticalResult',
                       'asr.plasmafrequency': 'PlasmaResult',
                       'asr.infraredpolarizability': 'InfraredResult',
                       'asr.bader': 'BaderResult',
                       'asr.bandstructure': 'BandStructureResult',
                       'asr.berry': 'BerryResult',
                       'asr.borncharges': 'BornChargesResult',
                       'asr.bse': 'BSEResult',
                       'asr.charge_neutrality': 'ChargeNeutralityResult',
                       'asr.chc': 'CHCResult',
                       'asr.convex_hull': 'ConvexHullResult',
                       'asr.defect_symmetry': 'DefectSymmetryResult',
                       'asr.defectformation': 'commented out',
                       'asr.defectinfo': 'DefectInfoResult',
                       'asr.defectlinks': 'DefectLinksResult',
                       'asr.deformationpotentials': 'DefPotsResult',
                       'asr.dimensionality': 'has none',
                       'asr.dos': 'DOSResult',
                       'asr.emasses': 'EmassesResult',
                       'asr.exchange': 'ExchangeResult',
                       'asr.fere': 'no web panel',
                       'asr.fermisurface': 'FermiSurfaceResult',
                       'asr.formalpolarization': 'none',
                       'asr.get_wfs': 'WfsResult',
                       'asr.gs': 'GsResult',
                       'asr.gw': 'GwResult',
                       'asr.hyperfine': 'HFResult',
                       'asr.hse': 'HSEResult',
                       'asr.magnetic_anisotropy': 'MagAniResult',
                       'asr.magstate': 'MagStateResult',
                       'asr.orbmag': 'OrbMagResult',
                       'asr.pdos': 'PDResult',
                       'asr.phonons': 'PhononResult',
                       'asr.phonopy': 'PhonopyResult',
                       'asr.piezoelectrictensor': 'PiezoEleTenResult',
                       'asr.projected_bandstructure': 'ProjBSResult',
                       'asr.raman': 'RamanResult',
                       'asr.relax': 'RelaxResult',
                       'asr.shg': 'ShgResult',
                       'asr.shift': 'ShiftResult',
                       'asr.sj_analyze': 'SJAnalyzeResult',
                       'asr.stiffness': 'StiffnessResult',
                       'asr.structureinfo': 'StructureInfoResult',
                       'asr.tdm': 'TdmResult',
                       'asr.zfs': 'ZfsResult',
                       }
        name = panel_names[module]
        mod = importlib.import_module(moved_module)
        result_handler = getattr(mod, name)
        return result_handler


def object_description_to_object(object_description: 'ObjectDescription'):
    """Instantiate object description."""
    return object_description.instantiate()


def data_to_dict(dct):
    """Recursively .todict all result instances."""
    for key, value in dct.items():
        if isinstance(value, ASRResult):
            dct[key] = value.todict()
            data_to_dict(dct[key])
    return dct


def dct_to_result(dct: dict) -> typing.Any:
    """Convert dict representing an ASR result to the corresponding result
    object."""
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
    else:
        newobj = obj
    return newobj


def decode_object(obj: typing.Any) -> typing.Any:
    """Convert the object representing an ASR result to the corresponding
    result object."""
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


def get_object_types(obj):
    """Get type hints of the object."""
    return typing.get_type_hints(obj)


def get_key_descriptions(obj):
    """Get key descriptions of the object."""
    if hasattr(obj, 'key_descriptions'):
        return obj.key_descriptions
    return {}


def format_key_description_pair(key: str, attr_type: type, description: str):
    """Format a key-type-description for a docstring.

    Parameters
    ----------
    key
        Key name.
    attr_type
        Key type.
    description
        Documentation of the key.
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
    """Prepare result class.

    This function reads the key descriptions and types defined in a Result
    class and assigns properties to all keys. It also sets strict=True used
    by the result object to ensure all data is present. It also changes
    the signature of the class to something more helpful than args, kwargs.
    """
    descriptions = get_key_descriptions(cls)
    types = get_object_types(cls)
    type_keys = set(types)
    description_keys = set(descriptions)
    missing_types = description_keys - type_keys
    assert not missing_types, (f'{cls.get_obj_id()}: '
                               f'Missing types for={missing_types}.')

    data_keys = description_keys
    for key in descriptions:
        description = descriptions[key]
        attr_type = types[key]
        setattr(cls, key, make_property(key, description,
                                        return_type=attr_type))

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
    cls.strict = True
    cls._known_data_keys = data_keys
    return cls


class ObjectDescription:
    """Result object descriptor."""

    def __init__(self, object_id: str, args: tuple, kwargs: dict,
                 constructor: typing.Optional[str] = None):
        """Initialize instance.

        Parameters
        ----------
        object_id: str
            ID of the object, e.g., 'asr.core.results::ASRResult' as
            produced by :py:func:`obj_to_id`.
        args
            Arguments for object construction.
        kwargs
            Keyword arguments for object construction.
        constructor: str or None
            ID of the constructor object, i.e., callable that can be used to
            instantiate the object. If unset, use constructor=object_id.
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
    resources={'time': 'a good time.'}
    >>> str(result)
    'a=1'
    >>> 'a' in result
    True
    >>> other_result = Result.fromdata(a=1)
    >>> result == other_result
    True
    >>> print(format(result, 'json'))
    {
     "object_id": "asr.core.results::Result",
     "constructor": "asr.core.results::Result",
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
      "strict": true
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
                 data: typing.Dict[str, typing.Any] = {},
                 metadata: typing.Dict[str, typing.Any] = {},
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
        self._data = data
        self._metadata = MetaData()
        self.metadata.set(**metadata)

        missing_keys = self.get_missing_keys()
        unknown_keys = self.get_unknown_keys()
        msg_ukwn = (f'{self.get_obj_id()}: '
                    f'Trying to set unknown keys={unknown_keys}')
        msg_miss = f'{self.get_obj_id()}: Missing data keys={missing_keys}'
        if 0:  # strict:
            assert not missing_keys, msg_miss
            assert not unknown_keys, msg_ukwn

    def get_missing_keys(self):
        data_keys = set(self.data)
        missing_keys = self._known_data_keys - data_keys
        return missing_keys

    def get_unknown_keys(self):
        data_keys = set(self.data)
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
            # constructor='asr.core::result_factory',
            args=(),
            kwargs={
                'data': self.data,
                'metadata': self.metadata,
                'strict': self.strict,
                # 'version': self.version,
            },
        )

    # To and from dict
    def todict(self):
        object_description = self.get_object_desc()
        return encode_object(object_description)

    @classmethod
    def fromdict(cls, dct: dict):
        obj_desc = ObjectDescription.fromdict(dct)
        return obj_desc.instantiate()

    # ---- Magic methods ----

    def copy(self):
        return self.data.copy()

    def __getitem__(self, item):
        """Get item from self.data."""
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        """Iterate over keys."""
        return iter(self.data)

    def __format__(self, fmt: str) -> str:
        """Encode result as string."""
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


class HackedASRResult(ASRResult):
    pass
