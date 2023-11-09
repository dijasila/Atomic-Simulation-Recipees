

class UnknownASRResultFormat(Exception):
    """Exception when encountering unknown results version number."""

    pass


class ModuleNameIsCorrupt(Exception):

    pass


class UnknownDataFormat(Exception):
    """Unknown ASR Result format."""

    pass


class MetaDataNotSetError(Exception):
    """Error raised when encountering an unknown metadata key."""

    pass


class CentroSymmetric(Exception):
    """CentroSymmetric crystals have vanishing SHG response!."""

    pass
