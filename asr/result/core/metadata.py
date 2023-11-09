import copy
from asr.result.core.exceptions import MetaDataNotSetError


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
        """Assert the integrity of metadata."""
        assert set(self._dct).issubset(self.accepted_keys)

    @property
    def asr_name(self):
        """For example, 'asr.gs.'"""
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

    def __contains__(self, key):
        """Is metadata key set."""
        return key in self._dct
