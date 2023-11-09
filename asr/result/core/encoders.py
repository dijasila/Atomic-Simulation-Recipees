from abc import ABC, abstractmethod

from ase.io import jsonio


class ResultEncoder(ABC):
    """Abstract encoder base class.

    Encodes a result object as a specific format. Optionally
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
    """Encoder for ASE compatible web panels."""

    def encode(self, result, row, key_descriptions):
        """Make basic webpanel.

        Simply print all attributes.
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
