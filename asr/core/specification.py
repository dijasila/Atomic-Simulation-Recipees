import copy
import typing
import uuid
from .params import Parameters
from .codes import Codes, Code
from .results import get_object_matching_obj_id
from .utils import only_master


@only_master
def get_new_uuid():
    return uuid.uuid4().hex


class RunSpecification:  # noqa

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

    def migrate(self):
        is_migrated = False
        migrated_data = {}
        for attr in [
                'name',
                'parameters',
                'codes',
                'version',
                'uid']:

            attribute = getattr(self, attr)
            if hasattr(attribute, 'migrate'):
                migrated = attr.migrate()
                if migrated:
                    is_migrated = True
                    migrated_data[attr] = migrated
                else:
                    migrated_data[attr] = attribute
            else:
                migrated_data[attr] = attribute
        if is_migrated:
            return RunSpecification(**migrated)

    def __format__(self, fmt):
        if fmt == '':
            return str(self)

    def __str__(self):  # noqa
        return (f'RunSpec(name={self.name}, params={self.parameters}, '
                f'version={self.version}, codes={self.codes}, uid={self.uid})')

    def __repr__(self):  # noqa
        return self.__str__()


def construct_run_spec(
        name: str,
        parameters: typing.Union[dict, Parameters],
        version: int = 0,
        codes: typing.Union[typing.List[str], Codes] = [],
        uid: str = None
) -> RunSpecification:
    """Construct a run specification."""
    if not isinstance(parameters, Parameters):
        parameters = Parameters(parameters)

    if not isinstance(codes, Codes):
        lst_codes = []
        for codestr in codes:
            try:
                code = Code.from_string(codestr)
                lst_codes.append(code)
            except ModuleNotFoundError:
                pass
        codes = Codes(lst_codes)

    if uid is None:
        uid = get_new_uuid()

    return RunSpecification(
        name=name,
        parameters=parameters,
        version=version,
        codes=codes,
        uid=uid,
    )
