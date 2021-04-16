import typing
import asr


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
    for name, value in record.parameters.items():
        if param_is_missing_gpaw_calculator(value):
            return True
    return False


@asr.migration(selector=select_records_that_misses_names_in_gpaw_calculator)
def add_missing_names_in_gpaw_calculators(record):
    """Add missing names in GPAW calculators."""
    for name, value in record.parameters.items():
        if param_is_missing_gpaw_calculator(value):
            value['name'] = 'gpaw'
    return record


custom_migrations = [
    add_missing_names_in_gpaw_calculators,
]
