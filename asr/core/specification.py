import copy
import typing
import uuid
from .params import Parameters
from .codes import Codes, Code
from .results import get_object_matching_obj_id


class RunSpecification:  # noqa

    spec_version: int = 0

    def __init__(  # noqa
            self,
            name: str,
            parameters: Parameters,
            version: int,
            codes: Codes,
            uid: str,
    ):
        self.name = name
        self.parameters = parameters
        self.codes = codes
        self.version = version
        self.uid = uid

    def __call__(  # noqa
            self,
            *args,
            **kwargs
    ):
        obj = get_object_matching_obj_id(self.name)
        function = obj.get_wrapped_function()
        parameters = copy.deepcopy(self.parameters)
        return function(*args, **kwargs, **parameters)

    def __str__(self):  # noqa
        return f'RunSpec(name={self.name}, params={self.parameters})'

    def __repr__(self):  # noqa
        return self.__str__()


def construct_run_spec(
        name: str,
        parameters: typing.Union[dict, Parameters],
        version: int,
        codes: typing.Union[typing.List[str], Codes] = [],
        uid: str = None
) -> RunSpecification:
    """Construct a run specification."""
    if not isinstance(parameters, Parameters):
        parameters = Parameters(parameters)

    if not isinstance(codes, Codes):
        codes = Codes([Code.from_string(code) for code in codes])

    if uid is None:
        uid = uuid.uuid4().hex

    return RunSpecification(
        name=name,
        parameters=parameters,
        version=version,
        codes=codes,
        uid=uid,
    )
