import textwrap
import copy
import typing
import uuid
import inspect
import importlib
from dataclasses import dataclass
from .parameters import Parameters
from .codes import Codes, Code
from .utils import only_master


@only_master
def get_new_uuid():
    return uuid.uuid4().hex


@dataclass
class RunSpecification:
    """Class that represents a run specification.

    A run specification is constructed when an instruction is run and represents
    all information available prior to the instruction has been executed.

    Attributes
    ----------
    name
        Name of the instruction.
    parameters: asr.Parameters
        Parameters passed to the instruction during execution.
    codes: asr.Codes
        Code versions.
    version: int
        Instruction version.
    uid
        Record UID.
    """

    name: str
    parameters: Parameters
    version: int
    codes: Codes
    uid: str
    # def __init__(  # noqa
    #         self,
    #         name: str,
    #         parameters: Parameters,
    #         version: int,
    #         codes: Codes,
    #         uid: str,
    # ):
    #     self.name = name
    #     self.parameters = parameters
    #     self.codes = codes
    #     self.version = version
    #     self.uid = uid

    def __call__(  # noqa
            self,
            *args,
            **kwargs
    ):
        obj = get_object_matching_obj_id(self.name)
        function = obj.get_wrapped_function()
        parameters = copy.deepcopy(self.parameters)
        return function(*args, **kwargs, **parameters)

    def __eq__(self, other):
        if not isinstance(other, RunSpecification):
            return False
        return self.__dict__ == other.__dict__

    def __format__(self, fmt):
        if fmt == '':
            return str(self)

    def __repr__(self):
        lines = []
        for key, value in sorted(self.__dict__.items(), key=lambda item: item[0]):
            value = str(value)
            if '\n' in value:
                value = '\n' + textwrap.indent(value, ' ')
            lines.append(f'{key}={value}')
        return '\n'.join(lines)


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


SEPARATOR = ':'


class ModuleNameIsCorrupt(Exception):

    pass


def get_object_matching_obj_id(asr_obj_id):
    module, name = asr_obj_id.split(SEPARATOR)
    if module in {'None.None', '__main__'}:
        raise ModuleNameIsCorrupt(
            """
            There is a problem with your result objectid module name={module}.
            This is a known bug. To fix the faulty result
            files please run:
            "python -m asr.utils.fix_object_ids folder1/
            folder2/ ..."
            where folder1 and folder2 are folders containing
            problematic result files."""
        )

    assert asr_obj_id.startswith(('asr.', 'ase.')),  \
        f'Invalid object id {asr_obj_id}'
    mod = importlib.import_module(module)
    cls = getattr(mod, name)

    assert cls
    return cls


def obj_to_id(cls):
    """Get a string representation of path to object.

    Ie. if obj is the ASRResult class living in the module, asr.core.results,
    the correspinding string would be 'asr.core.results::ASRResult'.

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

    return f'{package}.{modulename}{SEPARATOR}{objname}'
