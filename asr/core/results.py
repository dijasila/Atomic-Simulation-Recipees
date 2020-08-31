"""Implements ASRResults object and related quantities."""


class WebPanel:

    pass


class ASRResults:
    """Base class for describing results generated with recipes.

    WIP: Over time, by default, this class should be immutable.
    """

    def __init__(self, dct, webpanel: WebPanel = WebPanel):
        """Initialize results from dict."""
        self._check_dct(dct)
        self._dct = dct
        self._webpanel = webpanel

    def __getitem__(self, item):
        """Get item from self.dct."""
        return self.dct[item]

    def __contains__(self, item):
        """Determine if item in self.dct."""
        return item in self.dct

    def __iter__(self):
        """Iterate over keys."""
        return self.dct.__iter__()

    def __getattr__(self, key):
        """Get attribute."""
        return self.dct[key]

    def values(self):
        """Wrap self.dct.values."""
        return self.dct.values()

    def items(self):
        """Wrap self.dct.items."""
        return self.dct.items()

    def keys(self):
        """Wrap self.dct.keys."""
        return self.dct.keys()

    def webpanel(self):
        """Get web panel."""
        raise NotImplementedError


class GapResults(ASRResults):
    """Ground state results.

    Attributes
    ----------
    gap: float
        The band gap [eV].
    gap_dir: float
        The direct band gap [eV].
    """

    gap: float
    dipz: float


class GSResults(ASRResults):
    """Ground state results.

    Attributes
    ----------
    gap: float
        The band gap in eV.
    gaps_nosoc: GapResults
        Collection of band gap related results.
    """

    gap: float
    dipz: float
    gaps_nosoc: GapResults


print(GSResults)
print(GSResults.__dict__)
