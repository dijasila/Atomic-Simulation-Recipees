import typing
import asr
from asr.calculators import set_calculator_hook


def param_is_gpaw_calculator(param: typing.Any):
    if not isinstance(param, dict):
        return False
    if 'mode' in param:
        return True
    elif 'nbands' in param:
        return True
    return False


def param_is_missing_gpaw_calculator(param: typing.Any):
    if param_is_gpaw_calculator(param):
        if 'name' not in param:
            return True
    return False


def select_records_that_misses_names_in_gpaw_calculator(record):
    if record.version < 0:
        return False
    for _, value in record.parameters.items():
        if param_is_missing_gpaw_calculator(value):
            return True
    return False


@asr.mutation(
    selector=select_records_that_misses_names_in_gpaw_calculator,
    eagerness=5,
)
def add_missing_names_in_gpaw_calculators(record):
    """Add missing names in GPAW calculators."""
    for name, value in record.parameters.items():
        if param_is_missing_gpaw_calculator(value):
            value['name'] = 'gpaw'
    return record


def select_records_that_have_kpts_density_specified(record):
    if "atoms" not in record.parameters:
        return False
    # In some cases the input atoms were unknown
    # and resultfile.py assigns "Unknown atoms file" to these
    if record.parameters.atoms == "Unknown atoms file":
        return False
    for name, value in record.parameters.items():
        if 'calculator' in name and 'kpts' in value and 'density' in value['kpts']:
            return True
    return False


@asr.mutation(selector=select_records_that_have_kpts_density_specified, eagerness=-2)
def apply_calculator_hook_to_old_records(record):
    """Fix abstract calculator values to more concrete values."""
    parameters = set_calculator_hook(record.parameters)
    record.parameters = parameters
    return record


custom_mutations = [
    add_missing_names_in_gpaw_calculators,
    apply_calculator_hook_to_old_records,
]
